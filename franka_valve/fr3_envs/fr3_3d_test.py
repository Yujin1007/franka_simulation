from random import random, randint

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

import mujoco
from franka_valve.utils import rotations, tools

from franka_valve.fr3_envs.bases.fr3_rpy import Fr3_rpy

# Constants
BODY = 1
JOINT = 3
GEOM = 5
MOTION_TIME_CONST = 10.
TASK_SPACE_TIME = 3+1+0.5

RL = 2
MANUAL = 1
RPY = False

class Fr3_3d_test(Fr3_rpy):
    def __init__(self):
        super().__init__("frd_3d_test")

    def reset(self, direction=None):
        self.control_mode = 0
        self.episode_number += 1
        cnt_frame = 0
        done = False

        while not done and self.control_mode != 4:
            self.controller.initialize()
            self.data.qpos = self.q_init
            self.data.qvel = self.qdot_init
            self.direction = direction
            if self.episode_number % 2 == 0:
                r, obj, radius, init_angle = self.env_randomization()
                self.init_angle = init_angle
                self.r = r
                self.obj = obj
                self.radius = radius
            else:
                pass



            if self.direction == "clk":
                self.goal_angle = self.init_angle - 2 * np.pi
            elif self.direction == "cclk":
                self.goal_angle = self.init_angle + 2 * np.pi
            else:
                self.goal_angle = self.init_angle + 2 * np.pi * (-1) ** self.episode_number  # 2*np.pi
            self.episode_time = abs( MOTION_TIME_CONST * abs(self.goal_angle-self.init_angle) * self.radius) + TASK_SPACE_TIME
            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
            if self.episode_number % 2 == 0:
                self.controller.randomize_env(self.r, self.obj, self.init_angle, self.goal_angle, RL, RPY)
            else:
                self.controller.randomize_env(self.r, self.obj, self.init_angle, self.goal_angle, MANUAL, RPY)
            self.controller.control_mujoco()
            self.start_time = self.data.time


            self.time_done = False
            self.contact_done = False
            self.bound_done = False
            self.action_pre  = np.zeros(3)
            self.drpy_pre  = np.zeros(3)



            self.obs_q = np.zeros([self.stack, self.k])
            self.obs_6d = np.zeros([self.stack, 6])
            self.obs_rpy = np.zeros([self.stack,3])
            self.obs_rpy_des = np.zeros([self.stack,3])
            self.obs_xyz = np.zeros([self.stack, 3])
            self.obs_xyzdes = np.zeros([self.stack, 3])
            self.obs_manipulability = np.zeros([self.stack, 6])

            self.rpyfromvalve_data = []


            observation = dict(object=self.obs_object,q=self.obs_q, rpy=self.obs_rpy, rpy_des=self.obs_rpy_des, x_plan=self.obs_xyzdes,
                               x_pos=self.obs_xyz)

            # observation = dict(q=self.obs_q, rpy=self.obs_rpy, x_plan=self.obs_xyzdes, x_pos=self.obs_xyz,
            #                    manipulability=self.obs_manipulability)
            while self.control_mode != 4:
                self.control_mode = self.controller.control_mode()
                self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                     self.model.opt.timestep, self.data.xpos.reshape(66, ))
                self.controller.control_mujoco()
                self._torque, self.max_rotation = self.controller.write()
                for i in range(self.dof - 1):
                    self.data.ctrl[i] = self._torque[i]

                mujoco.mj_step(self.model, self.data)
                done = self._done()
                if done:
                    break
                if cnt_frame == 100:
                    cnt_frame = 0
                    end_effector = self.controller.get_ee()
                    self.save_frame_data(end_effector)
                cnt_frame += 1
                if self.rendering:
                    self.render()
                    # sleep(0.002)


        return observation

    def step(self, action):
        # print("action  : ", action)
        # drpy = tools.orientation_6d_to_euler(action)
        # print(drpy)
        drpy = action
        self.run_step(drpy)

        ee = self.controller.get_ee()
        rpy_des = self.controller.desired_rpy()
        obs = self._observation(ee,rpy_des)
        done = self._done()
        reward = self._reward(action)
        info = self._info()

        self.drpy_pre = drpy
        self.action_pre = action

        return obs, reward, done, info
    
    def _reward(self,action):
        reward_acc = np.exp(-sum(abs(action - self.action_pre))) #max = -12 * const
        reward_xyz = np.exp(-2*sum(abs(self.obs_xyz[0] - self.obs_xyzdes[0])))
        reward_rpy = np.exp(-2 * sum(abs(rotations.subtract_euler(self.obs_rpy_des[0], self.obs_rpy[0]))))

        reward = self.update_reward(reward_acc, reward_xyz, reward_rpy)
        return reward

    def env_randomization(self):
        obj_list = ["handle", "valve"]
        radius_list = [0.119, 0.1]
        o = randint(0,1)
        o = 0
        obj = obj_list[o]
        radius = radius_list[o]

        quat_candidate, pos_candidate, nobj = self.read_candidate_json(obj, "candidate_rpy.json")
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

            self.model.body_quat[bid] = random_quat
            self.model.body_pos[bid] = random_pos
            self.model.body_pos[nbid] += 3
            r = R.from_quat(tools.quat2xyzw(random_quat))


        else:

            i = self.episode_number if self.episode_number <= 6 else self.episode_number - 7

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

        # bring init angle

        if self.direction == "clk":
            classifier = self.classifier_clk
            add_idx = -1
            direction = -1
        elif self.direction == "cclk":
            classifier = self.classifier_cclk
            add_idx = +1
            direction = +1
        if obj == "handle":
            obj_id = [1,0,1]
            input_data = random_quat + random_pos + [radius]
            test_input_data = torch.Tensor(input_data).cuda()
            predictions = classifier(test_input_data)
            result = torch.argmax(predictions)
            result = result.item()
            self.o_margin = [[0], [0.149], [0]]
            self.T_vv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            if result in [2, 11, 20, 29]:
                result = result + 2 * add_idx
            elif result in [1, 10,  19,  28]:
                result = result + 2*add_idx

        elif obj == "valve":
            obj_id = [0, 1, 0]
            result = 0
            self.o_margin = [[0], [0], [-0.017]]
            self.T_vv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        init_angle = 2*np.pi*result/36
        # self.obs_object = np.concatenate([self.model.body_pos[bid], obj_rotation6d, [result/36], [direction]], axis=0)
        self.obs_object = np.concatenate([self.model.body_pos[bid], obj_rotation6d, [result / 36], [direction], obj_id],
                                         axis=0)

        # self.obs_object = self.obs_object.reshape((1, 11))
        self.obs_object = self.obs_object.reshape((1, 14))

        # print("now :", i, "  direction", direction, "  grasp : ",result)


        self.obj_pos = random_pos
        self.obj_rotation = r.as_matrix()
        self.normal_vector = self.obj_rotation @ self.o_margin
        self.obj_normal = [0, 0, 0]
        for idx in range(3):
            self.obj_normal[idx] = self.normal_vector[idx][0] + self.obj_pos[idx]

        return r.as_matrix().tolist(), obj, radius, init_angle