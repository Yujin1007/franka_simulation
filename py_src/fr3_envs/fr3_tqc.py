import numpy as np

import mujoco
from gym import spaces
from random import random, randint
from scipy.spatial.transform import Rotation as R
from utils import tools, rotations
import torch

from fr3_envs.bases.fr3_rpy import Fr3_rpy

# Constants
BODY = 1
JOINT = 3
GEOM = 5
MOTION_TIME_CONST = 10.
TASK_SPACE_TIME = 3+1+0.5

RL = 2
MANUAL = 1

RPY = False
XYZRPY = True

JOINT_CONTROL = 1
TASK_CONTROL = 2
CIRCULAR_CONTROL = 3
RL_CIRCULAR_CONTROL = 4
RL_CONTROL = 6

class Fr3_tqc(Fr3_rpy):
    def __init__(self):
        super().__init__("fr3_tqc")

    def reset(self, direction=None):
        self.control_mode = 0

        self.direction = direction
        cnt_frame = 0
        env_reset = True
        self.episode_time_elapsed = 0.0
        self.handle_angle = 0.0
        self.handle_angle_reset = 0.0
        self.action_reset = 0
        self.cnt_reset = 0

        while env_reset:
            self.episode_number += 1
            self.start_time = self.data.time + 1
            self.controller.initialize()
            self.data.qpos = self.q_init
            self.data.qvel = self.qdot_init

            if self.direction is None:
                self.direction = self.direction_state[randint(0, 1)]

            r, obj, radius, init_angle = self.env_randomization() #self.obs_object initialize
            self.init_angle = init_angle

            if self.direction == "clk":
                self.goal_angle = init_angle - 5*np.pi
            elif self.direction == "cclk":
                self.goal_angle = init_angle + 5*np.pi

            self.required_angle = abs(self.goal_angle - self.init_angle)
            self.episode_time = abs( MOTION_TIME_CONST * self.required_angle * radius)

            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos[:22].reshape(66, ))
            self.controller.randomize_env(r, obj, self.data.xpos[:22].reshape(66, ), self.init_angle, self.goal_angle, RL, RPY)
            self.controller.control_mujoco()

            self.contact_done = False
            self.bound_done = False
            self.goal_done = False
            self.reset_done = False
            self.action_pre  = np.zeros(6)
            self.drpy_pre  = np.zeros(3)

            self.obs_q = np.zeros([self.stack, self.k])
            self.obs_rpy = np.zeros([self.stack, 6])
            self.obs_xyz = np.zeros([self.stack, 3])

            self.rpyfromvalve_data = []
            self.path_data = []

            while self.control_mode != 4:
                self.control_mode = self.controller.control_mode()
                self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                     self.model.opt.timestep, self.data.xpos[:22].reshape(66, ))
                self.controller.control_mujoco()
                self._torque, self.max_rotation = self.controller.write()
                for i in range(self.dof - 1):
                    self.data.ctrl[i] = self._torque[i]

                mujoco.mj_step(self.model, self.data)
                ee = self.controller.get_ee()

                self.path_data.append([ee[1] + self.data.qpos[:7].tolist()])
                done = self._done()
                normalized_q = self.obs_q[0]
                if done or max(abs(normalized_q)) > 0.95:
                    break

                # --- collect observation for initialization ---
                if cnt_frame == 100:
                    cnt_frame = 0
                    end_effector = self.controller.get_ee()
                    # self.save_frame_data(end_effector)
                    obs = self._observation(end_effector)

                cnt_frame += 1
                self.render_mujoco()

            # control mode : 4
            env_reset = False
            self.start_time = self.data.time
            self.q_reset[:self.k] = self.data.qpos[:self.k]

        return obs

    def step(self, action_rotation):
        drpy = tools.orientation_6d_to_euler(action_rotation)
        normalized_q = self.obs_q[0]

        if max(abs(normalized_q)) > 0.95:
            self.action_reset = 1
            self.cnt_reset += 1
            # print(self.cnt_reset, end="|")
            # if self.cnt_reset >= 10:
            #     self.rendering = True
        else:
            self.action_reset = 0

        if self.action_reset:
            self.handle_angle_reset += max(abs(self.data.qpos[-2:]))
            self.data.qpos= self.q_reset
            self.data.qvel = self.qdot_init

            mujoco.mj_step(self.model, self.data)
            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos[:22].reshape(66, ))
            self.controller.target_replan()
            self.render_mujoco()
        else:
            done = self.control_mujoco(drpy)

        ee = self.controller.get_ee()
        obs = self._observation(ee)
        done = self._done()
        reward = self._reward(action_rotation, done)
        info = self._info()

        self.drpy_pre = drpy
        self.action_pre = action_rotation

        return obs, reward, done, info
    
    def render_mujoco(self):
        if self.rendering:
            self.render()

    def control_mujoco(self, drpy):
        done = False
        duration = 0

        while not done:
            done = self._done()
            self.control_mode = self.controller.control_mode()
            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos[:22].reshape(66, ))

            # --- RL controller input ---
            if self.control_mode == RL_CIRCULAR_CONTROL:
                drpy_tmp = (drpy - self.drpy_pre) / 100 * duration + self.drpy_pre
                duration += 1
                self.controller.put_action(drpy_tmp)

            if duration == 100:
                self.drpy_pre = drpy
                break

            self.controller.control_mujoco()
            self._torque, self.max_rotation = self.controller.write()

            for i in range(self.dof-1):
                self.data.ctrl[i] = self._torque[i]

            mujoco.mj_step(self.model, self.data)
            self.render_mujoco()
        
        return done

    def _observation(self, end_effector):
        # stack observations
        self.obs_q[1:] = self.obs_q[:-1]
        self.obs_xyz[1:] = self.obs_xyz[:-1]
        self.obs_rpy[1:] = self.obs_rpy[:-1]

        q_unscaled = self.data.qpos[0:self.k]
        q = (q_unscaled-self.q_range[:,0]) / (self.q_range[:,1] - self.q_range[:,0]) * (1-(-1)) - 1
        dq_unscaled = self.data.qvel[0:self.k]
        dq = (dq_unscaled - self.qdot_range[:, 0]) / (self.qdot_range[:, 1] - self.qdot_range[:, 0]) * (1 - (-1)) - 1

        xyz = end_effector[1][:3]
        rpy = end_effector[1][3:6]
        r6d = tools.orientation_euler_to_6d(rpy)

        self.obs_xyz[0] = xyz
        self.obs_rpy[0] = r6d
        self.obs_q[0] = q

        observation = dict(object=self.obs_object,q=self.obs_q,rpy=self.obs_rpy, x_pos=self.obs_xyz)
        # self.save_frame_data(end_effector)
        # 이 부분 차이점
        observation = self._flatten_obs(observation)

        return observation
    
    def _reward(self, action, done):
        if (self.action_pre == 0.0).all():
            self.action_pre = action
        reward_acc = -sum(abs(rotations.subtract_euler(tools.orientation_6d_to_euler(action),
                                                                  tools.orientation_6d_to_euler(self.action_pre))))

        reward_grasp = 0
        reward_contact = 0
        reward_bound = 0

        if self.control_mode == RL_CIRCULAR_CONTROL: # 잡은 이후
            if not -1 in self.contact_list:
                reward_grasp = -2+len(self.grasp_list) # grasp_list max = 8 : finger parts.
        if self.action_reset:
            reward_bound = -1
        if done:
            if self.contact_done:
                reward_contact = -1


        reward = self.rw_acc*reward_acc\
                 +self.rw_gr*reward_grasp\
                 +self.rw_c*reward_contact\
                 +self.rw_b*reward_bound

        return reward


    def _done(self):
        mid_result = super()._done()

        self.reset_done = self.cnt_reset >= 5
        self.handle_angle = max(abs(self.data.qpos[-2:])) + self.handle_angle_reset
        self.goal_done = abs(self.required_angle - self.handle_angle) < 0.01

        return True if mid_result or self.goal_done or self.reset_done else False

    def _construct_observation_space(self):
        s = {
            'object': spaces.Box(shape=(1, 13), low=-np.inf, high=np.inf, dtype=np.float32),
            'q': spaces.Box(shape=(self.stack, self.k), low=-1, high=1, dtype=np.float32),
            'rpy': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'x_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
        }
        observation = spaces.Dict(s)
        observation.shape = 0
        for _, v in s.items():
            observation.shape += v.shape[0] * v.shape[1]
        return observation

    def _flatten_obs(self, observation):
        flatten_obs = []
        for k,v in observation.items():
            flatten_obs = np.concatenate([flatten_obs, v.flatten()])
        return flatten_obs
    
    def env_randomization(self):

        obj_list = ["handle", "valve"]
        radius_list = [0.119, 0.1]
        o = randint(0,1)
        o = 0 #valve 대상으로 한 코드는 아직 (x)
        obj = obj_list[o]
        radius = radius_list[o]

        quat_candidate, pos_candidate, nobj = self.read_candidate_json(obj, "candidate_tqc.json")
        bid = mujoco.mj_name2id(self.model, BODY, obj)
        nbid = mujoco.mj_name2id(self.model, BODY, nobj)

        if self.env_rand:
            i = randint(0, 6)
            axis = ['x', 'y', 'z']

            add_quat = R.from_euler(axis[randint(0, 2)], (random() - 0.5))
            ori_quat = R.from_quat(tools.quat2xyzw(quat_candidate[i]))
            new_quat = add_quat * ori_quat
            random_quat = tools.xyzw2quat(new_quat.as_quat()).tolist()

            add_pos = [(random() - 0.5) / 5, (random() - 0.5) / 5, (random() - 0.5) / 5]
            random_pos = [x + y for x, y in zip(add_pos, pos_candidate[i])]
            # random_pos = [(random() * 0.4 + 0.3), (random()*0.8 - 0.4), random() * 0.7 + 0.1]
            self.model.body_quat[bid] = random_quat
            self.model.body_pos[bid] = random_pos
            # print("quat:",random_quat, "pos: ",random_pos)
            self.model.body_pos[nbid] += 3
            r = R.from_quat(tools.quat2xyzw(random_quat))


        else:
            i = self.episode_number if self.episode_number <= 6 else self.episode_number - 7
            # print(i)
            # i = 4
            self.direction = "cclk"
            random_quat = quat_candidate[i]
            random_pos = pos_candidate[i]
            self.model.body_quat[bid] = random_quat
            self.model.body_pos[bid] = random_pos
            self.model.body_pos[nbid] += 3
            r = R.from_quat(tools.quat2xyzw(random_quat))
            # random_quat = self.model.body_quat[bid].copy().tolist()
            # random_pos =  self.model.body_pos[bid].copy().tolist()
            # r = R.from_quat(tools.quat2xyzw(random_quat))

        mujoco.mj_step(self.model, self.data)
        self.obj = obj
        obj_rotation6d = tools.orientation_quat_to_6d(self.model.body_quat[bid], "mujoco")

        if self.direction == "clk":
            classifier = self.classifier_clk
            direction = -1
        elif self.direction == "cclk":
            classifier = self.classifier_cclk
            direction = +1

        if obj == "handle":
            obj_id = [1,0,1]
            input_data = random_quat + random_pos
            test_input_data = torch.Tensor(input_data).cuda()
            predictions = classifier(test_input_data)
            angles = [4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 22, 23, 24, 25, 26, 31, 32, 33, 34, 35]
            result = torch.argmax(predictions)
            result = angles[result]

            self.o_margin = [[0], [0.149], [0]]
            self.T_vv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            # print("direction :", self.direction, "input:",input_data)
            # print("result :", torch.argmax(predictions), "angles :", result, "output:",predictions)

        elif obj == "valve":
            obj_id = [0, 1, 0]
            result = 0
            self.o_margin = [[0], [0], [-0.017]]
            self.T_vv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        # result = 17
        init_angle = 2*np.pi*result/36
        self.obs_object = np.concatenate([self.model.body_pos[bid], obj_rotation6d, [direction], obj_id],
                                         axis=0)
        self.obs_object = self.obs_object.reshape((1, 13))


        self.obj_pos = random_pos
        self.obj_rotation = r.as_matrix()
        self.normal_vector = self.obj_rotation @ self.o_margin
        self.obj_normal = [0, 0, 0]
        for idx in range(3):
            self.obj_normal[idx] = self.normal_vector[idx][0] + self.obj_pos[idx]

        return r.as_matrix().tolist(), obj, radius, init_angle
