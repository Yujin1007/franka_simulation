import os
import torch
from abc import *
import numpy as np

from utils import tools
import mujoco
from mujoco import viewer
import gym
from gym import spaces

from fr3_envs.bases.fr3 import Fr3
from models.classifier.classifier_tqc import Classifier
from models.tqc import DEVICE
HOME = os.getcwd()

# Constants
BODY = 1
JOINT = 3
GEOM = 5
MOTION_TIME_CONST = 10.
TASK_SPACE_TIME = 3+1+0.5

RL = 2
MANUAL = 1

class Fr3_rpy(Fr3):
    def __init__(self, env, rw_acc, rw_c, rw_b, rw_gr, history):
        super().__init__(history)
        self.env = env

        self.observation_space = self._construct_observation_space()
        self.action_space = self._construct_action_space()

        ## reward weight
        self.reward_range = None
        self.rw_acc = rw_acc # reward
        self.rw_c = rw_c  # penalty
        self.rw_b = rw_b   # penalty
        self.rw_gr = rw_gr # reward

        self.viewer = None
        self.env_rand = False
        self.q_range = self.model.jnt_range[:self.k]

        self.qdot_init = [0,0,0,0,0,0,0,0,0,0,0]
        self.q_init = [0.374, -1.02, 0.245, -1.51, 0.0102, 0.655, 0.3, 0.04, 0.04, 0, 0]
        self.episode_number = -1

        os.chdir(HOME)
        self.classifier_clk = self.BringClassifier(os.path.join("models", "classifier", "model_clk.pt"))
        self.classifier_cclk = self.BringClassifier(os.path.join("models", "classifier", "model_cclk.pt"))

        desired_contact_list, desired_contact_list_finger, desired_contact_list_obj,\
        robot_contact_list, object_contact_list = self.read_contact_json("contact_tqc.json" if env == "fr3_tqc" else "contact_rpy.json")

        self.desired_contact_bid = tools.name2id(self.model, GEOM, desired_contact_list)
        self.desired_contact_finger_bid = tools.name2id(self.model, GEOM, desired_contact_list_finger)
        self.desired_contact_obj_bid = tools.name2id(self.model, GEOM, desired_contact_list_obj)
        self.robot_contact_bid = tools.name2id(self.model, GEOM, robot_contact_list)
        self.object_contact_bid = tools.name2id(self.model, GEOM, object_contact_list)

        # Environments with custom TQC
        if env == "fr3_tqc":
            self.direction_state = ["clk","cclk"]
            self.qdot_range = np.array([[-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750], [-2.1750, 2.1750],
                                        [-2.61, 2.61], [-2.61, 2.61], [-2.61, 2.61]])
            self.q_init = [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45),0,0,0,0]
            self.q_reset = [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45),0.04,0.04,0,0]
        # Environments with stable_baselines
        else:
            self.rw_xyz = 0.1  # np.exp(-2*sum(abs(self.obs_xyz[0] - self.obs_xyzdes[0])))
            self.rw_t = 1  # time done -> reward_time = 1
            self.rw_rpy = 0.0  # np.exp(-2*sum(abs(rotations.subtract_euler(self.obs_rpy_des, self.obs_rpy))[0]))

            self.ADR_threshold = 20
            self.ADR_cnt = 0
            self.ADR_object = 1

    def BringClassifier(self, path):
        classifier = Classifier(input_size=7, output_size=20).to(DEVICE)
        classifier.load_state_dict(torch.load(path, map_location=DEVICE))
        classifier.eval()
        return classifier

    def run(self, iteration):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        iter = 0
        cnt = 0
        self.done = False
        self.reset()
        while self.viewer.is_running():
            if not self.done:
                self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                     self.model.opt.timestep, self.data.xpos.reshape(66,))
                self.controller.put_action([self.dr[cnt], self.dp[cnt], self.dy[cnt]])
                # print("python handdot :", self.dr[cnt],",",self.dp[cnt],", ", self.dy[cnt])
                self.controller.control_mujoco()
                self._torque, self.max_rotation = self.controller.write()
                for i in range(self.dof-1):
                    self.data.ctrl[i] = self._torque[i]

                mujoco.mj_step(self.model, self.data)

                ee = self.controller.get_ee()
                obs = self._observation(ee)
                self.done = self._done()
                self.viewer.sync()
                cnt +=1
                # sleep(0.002)
            else:
                self.reset()

                iter += 1
                self.done = False
            if iter == iteration:
               self.viewer.close()

    # Called by step()
    def run_step(self, drpy):
        while self.control_mode != 4:
            self.step_mujoco()
            done = self._done()
            if done:
                break
            if self.rendering:
                self.render()
                # sleep(0.002)

        for j in range(100): # 1000hz -> 10hz
            self.step_mujoco_control4(j, drpy)
            done = self._done()
            if done:
                break
            if self.rendering:
                self.render()

    def step_mujoco(self):
        self.control_mode = self.controller.control_mode()
        # self.control_mode = 0
        self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                            self.model.opt.timestep, self.data.xpos.reshape(66, ))
        self.controller.control_mujoco()
        self._torque, self.max_rotation = self.controller.write()
        for i in range(self.dof-1):
            self.data.ctrl[i] = self._torque[i]

        mujoco.mj_step(self.model, self.data)

    def step_mujoco_control4(self, j, drpy):
        self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
        if j < 10:
            drpy_tmp = (drpy - self.drpy_pre)/10 * j + self.drpy_pre
            self.controller.put_action(drpy_tmp)
        else:
            self.controller.put_action(drpy)

        # calculate x_des_hand.head(3) and generate control torque
        self.controller.control_mujoco()

        # take action
        self._torque, self.max_rotation = self.controller.write()
        for i in range(self.dof-1):
            self.data.ctrl[i] = self._torque[i]

        mujoco.mj_step(self.model, self.data)

    def _observation(self, end_effector, rpy_des):
        # stack observations
        self.obs_q[1:] = self.obs_q[:-1]
        self.obs_xyzdes[1:] = self.obs_xyzdes[:-1]
        self.obs_xyz[1:] = self.obs_xyz[:-1]
        self.obs_rpy[1:] = self.obs_rpy[:-1]
        self.obs_rpy_des[1:] = self.obs_rpy_des[:-1]

        q_unscaled = self.data.qpos[0:self.k]
        q = (q_unscaled-self.q_range[:,0]) / (self.q_range[:,1] - self.q_range[:,0]) * (1-(-1)) - 1
        self.obs_q[0] = q
        rpy = end_effector[1][3:6]
        jacobian = np.array(end_effector[2])

        self.obs_xyzdes[0] = end_effector[0]
        self.obs_xyz[0] = end_effector[1][0:3]
        
        self.obs_rpy[0] = tools.orientation_euler_to_6d(rpy) if self.env[:2] == "6d" else rpy
        self.obs_rpy_des[0] = tools.orientation_euler_to_6d(rpy_des) if self.env[:2] == "6d" else rpy

        self.obs_manipulability[0] = tools.calc_manipulability(jacobian)
        observation = dict(object=self.obs_object,q=self.obs_q,rpy=self.obs_rpy, rpy_des=self.obs_rpy_des, x_plan=self.obs_xyzdes, x_pos=self.obs_xyz)
        
        self.save_frame_data(end_effector)
        return observation

    def update_reward(self, reward_acc, reward_xyz, reward_rpy):
        reward_time = 0
        reward_grasp = 0
        reward_contact = 0
        reward_bound = 0

        if self.time_done:
            reward_time = 1

        if self.data.time - self.start_time >= TASK_SPACE_TIME: # 잡은 이후
            if not -1 in self.contact_list:
                reward_grasp = -2+len(self.grasp_list) # grasp_list max = 8 : finger parts.
        if self.contact_done:
            reward_contact = -1
        if self.bound_done:
            reward_bound = -1

        reward = self.rw_acc*reward_acc\
                 +self.rw_xyz*reward_xyz\
                 +self.rw_t*reward_time\
                 +self.rw_gr*reward_grasp\
                 +self.rw_c*reward_contact\
                 +self.rw_b*reward_bound\
                 +self.rw_rpy*reward_rpy

        return reward

    def _done(self):
        self.contact_list = tools.detect_contact(self.data.contact, self.desired_contact_bid)
        self.grasp_list = tools.detect_grasp(self.data.contact, self.obj, self.desired_contact_finger_bid, self.desired_contact_obj_bid)
        self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)

        self.time_done = self.data.time - self.start_time >= self.episode_time
        self.contact_done = -1 in self.contact_list
        self.bound_done = -1 in self.q_operation_list
        
        return True if self.time_done or self.contact_done or self.bound_done else False

    def _construct_action_space(self):
        action_space = 6
        action_low = -1*np.ones(action_space)
        action_high = 1* np.ones(action_space)
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    
    def _construct_observation_space(self):
        s = {'object': spaces.Box(shape=(1, 14), low=-np.inf, high=np.inf, dtype=np.float32),
             'q': spaces.Box(shape=(self.stack, self.k), low=-1, high=1, dtype=np.float32),
             'rpy': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
             'rpy_des': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
             'x_plan': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
             'x_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
            }
        return spaces.Dict(s)   