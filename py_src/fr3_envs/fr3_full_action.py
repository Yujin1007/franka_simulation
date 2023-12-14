import os
import json
from abc import *
from random import random, randint

import numpy as np
from scipy.spatial.transform import Rotation as R

from utils import tools
import mujoco
import gym
from gym import spaces

import fr3_envs.fr3 as fr3

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

HOME = os.getcwd()

class Fr3_full_action(fr3.Fr3):
    def __init__(self):        
        super().__init__()

        self.reward_accum = 0
        self.observation_space = self._construct_observation_space()
        self.action_space = self._construct_action_space()

        ## reward weight
        self.reward_range = None
        self.episode_time = 10.0

        self.rw_time = 10
        self.rw_contact = 0
        self.rw_goal = 10
        self.rw_bound = 10  # joint boundary done -> -1
        self.rw_grasp = .25  # 1/-1 grasp
        self.rw_distance = 0.5
        self.rw_rotation = 5
        self.rw_velocity = 0.01

        self.viewer = None
        self.env_rand = False
        self.q_range = self.model.jnt_range[:self.dof]
        self.qdot_range = np.array([[-2.1750,2.1750],[-2.1750,2.1750],[-2.1750,2.1750],[-2.1750,2.1750],
                                    [-2.61,2.61],[-2.61,2.61],[-2.61,2.61],[-0.1,0.1],[-0.1,0.1]])
        self.q_init =  [0, np.deg2rad(-60), 0, np.deg2rad(-90), 0, np.deg2rad(90), np.deg2rad(45),0,0,0,0]
        self.qdot_init = [0,0,0,0,0,0,0,0,0,0,0]
        # self.q_init = [0, -0.7853981633974483 ,0,-2.356194490192345,0,1.5707963267948966,0.7853981633974483,0,0,0,0]
        # self.read_file()
        self.episode_number = 0

        desired_contact_list, desired_contact_list_finger, desired_contact_list_obj,\
        robot_contact_list, object_contact_list = self.read_contact_json("contact_full_action.json")

        self.desired_contact_bid = tools.name2id(self.model, GEOM, desired_contact_list)
        self.desired_contact_finger_bid = tools.name2id(self.model, GEOM, desired_contact_list_finger)
        self.desired_contact_obj_bid = tools.name2id(self.model, GEOM, desired_contact_list_obj)
        self.robot_contact_bid = tools.name2id(self.model, GEOM, robot_contact_list)
        self.object_contact_bid = tools.name2id(self.model, GEOM, object_contact_list)

        self.T_vv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

    def reset(self):
        self.controller.initialize()
        self.data.qpos = self.q_init
        self.data.qvel = self.qdot_init
        self.handle_angle = 0
        self.gripper_pre = 0
        r, obj, radius, init_angle = self.env_randomization()  # self.obs_object initialize
        self.radius = radius
        self.obj = obj
        self.init_angle = init_angle # <- 0
        self.goal_angle = init_angle + 4 * np.pi#*(-1)**self.episode_number   # <- +/- 2*np.pi
        # self.goal_angle = -4 * np.pi
        self.rotation_angle_pre = self.goal_angle - self.init_angle

        self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                             self.model.opt.timestep, self.data.xpos.reshape(66, ))
        
        self.controller.randomize_env(r, obj, self.data.xpos[:22].reshape(66, ), self.init_angle, self.goal_angle, RL, XYZRPY)
        self.controller.control_mujoco()
        self.start_time = self.data.time

        self.time_done = False
        self.contact_done = False
        self.bound_done = False
        self.goal_done = False

        self.dxyz_pre = np.zeros(3)
        self.drpy_pre = np.zeros(3)
        self.obs_q = np.zeros([self.stack, self.dof])
        self.obs_dq = np.zeros([self.stack, self.dof])
        self.obs_6d = np.zeros([self.stack, 6])
        self.obs_xyz = np.zeros([self.stack, 3])
        self.obs_6d_dot = np.zeros([self.stack, 6])
        self.obs_xyz_dot = np.zeros([self.stack, 3])
        self.obs_object = np.zeros([1, 17])

        end_effector = self.controller.get_ee()
        # end_effector = np.zeros((2,6))
        _ = self._observation(end_effector)
        self.distance_init = np.sqrt(
            (end_effector[1][0] - self.obj_normal[0]) ** 2 + (end_effector[1][1] - self.obj_normal[1]) ** 2 + (
                        end_effector[1][2] - self.obj_normal[2]) ** 2)


        for i in range(1,self.stack):
            self.obs_q[i] = self.obs_q[0]
            self.obs_dq[i] = self.obs_dq[0]
            self.obs_6d[i] = self.obs_6d[0]
            self.obs_xyz[i] = self.obs_xyz[0]
            self.obs_6d_dot[i] = self.obs_6d_dot[0]
            self.obs_xyz_dot[i] = self.obs_xyz_dot[0]


        observation = dict(object=self.obs_object,q=self.obs_q,
                           r6d=self.obs_6d, x_pos=self.obs_xyz)
        self.episode_number += 1

        self.render()
        return observation

    def step(self, action):
        # random_data1 = [(random()-0.5)*3,(random()-0.5)*3,(random()-0.5)*3]
        # random_data2 = [random()-0.5,random()-0.5,random()-0.5]

        drpy = tools.orientation_6d_to_euler(action[:6])
        dxyz = action[6:9]

        if action[9] > 0:
            self.gripper = 0.04
        else:
            self.gripper = 0.01
        # self.gripper = action[9]
        # print(self.gripper)
        for j in range(100): # 1Khz -> 10hz

            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))

            if j<10:
                drpy_tmp = (drpy - self.drpy_pre)/10 * j + self.drpy_pre
                dxyz_tmp = (drpy - self.dxyz_pre)/10 * j + self.dxyz_pre
                self.controller.put_action2(drpy_tmp,dxyz_tmp,self.gripper)
            else:
                self.controller.put_action2(drpy, dxyz, self.gripper)


            self.controller.control_mujoco()

            self._torque, self.max_rotation = self.controller.write()
            for i in range(self.dof-1):
                self.data.ctrl[i] = self._torque[i]

            mujoco.mj_step(self.model, self.data)
            done = self._done()
            if done:
                break
            if self.rendering:
                self.render()


        ee = self.controller.get_ee()
        obs = self._observation(ee)
        done = self._done()
        reward = self._reward(action)
        self.reward_accum += reward

        info = self._info()

        self.drpy_pre = drpy
        self.dxyz_pre = dxyz
        self.gripper_pre = self.gripper

        return obs, reward, done, info

    def _observation(self, end_effector):
        # stack observations

        self.obs_q[1:] = self.obs_q[:-1]
        self.obs_6d[1:] = self.obs_6d[:-1]
        self.obs_xyz[1:] = self.obs_xyz[:-1]
        self.obs_dq[1:] = self.obs_dq[:-1]
        self.obs_6d_dot[1:] = self.obs_6d_dot[:-1]
        self.obs_xyz_dot[1:] = self.obs_xyz_dot[:-1]

        q_unscaled = self.data.qpos[0:self.dof]
        q = (q_unscaled-self.q_range[:,0]) / (self.q_range[:,1] - self.q_range[:,0]) * (1-(-1)) - 1
        self.obs_q[0] = q

        qdot_unscaled = self.data.qvel[0:self.dof]
        qdot = (qdot_unscaled - self.qdot_range[:, 0]) / (self.qdot_range[:, 1] - self.qdot_range[:, 0]) * (1 - (-1)) - 1
        self.obs_dq[0] = qdot

        r = R.from_euler('xyz', end_effector[1][3:6], degrees=False)
        r6d = tools.orientation_quat_to_6d(r.as_quat(), 'scipy')
        r = R.from_euler('xyz', end_effector[0][3:6], degrees=False)
        r6d_dot = tools.orientation_quat_to_6d(r.as_quat(), 'scipy')


        self.obs_6d[0] = r6d
        self.obs_xyz[0] = end_effector[1][0:3]

        self.obs_6d_dot[0] = r6d_dot
        self.obs_xyz_dot[0] = end_effector[0][0:3]

        if self.obj == "handle":
            valve_theta = self.data.qpos[10] - 0.225 + np.pi*0.25
            p1 = self.obj_rotation @ [[0.119*np.cos(valve_theta + np.pi*0.0)],[0.149],[0.119*np.sin(valve_theta + np.pi*0.0)]]
            p2 = self.obj_rotation @ [[0.119*np.cos(valve_theta + np.pi*0.5)],[0.149],[0.119*np.sin(valve_theta + np.pi*0.5)]]
            p3 = self.obj_rotation @ [[0.119*np.cos(valve_theta + np.pi*1.0)],[0.149],[0.119*np.sin(valve_theta + np.pi*1.0)]]
            p4 = self.obj_rotation @ [[0.119*np.cos(valve_theta + np.pi*1.5)],[0.149],[0.119*np.sin(valve_theta + np.pi*1.5)]]


            obj_p1 = [0,0,0]
            obj_p2 = [0,0,0]
            obj_p3 = [0,0,0]
            obj_p4 = [0,0,0]
            for idx in range(3):
                obj_p1[idx] = p1[idx][0] + self.obj_pos[idx] +np.random.normal(0,0.05,1)[0]
                obj_p2[idx] = p2[idx][0] + self.obj_pos[idx]+np.random.normal(0,0.05,1)[0]
                obj_p3[idx] = p3[idx][0] + self.obj_pos[idx]+np.random.normal(0,0.05,1)[0]
                obj_p4[idx] = p4[idx][0] + self.obj_pos[idx]+np.random.normal(0,0.05,1)[0]


            # current theta, goal theta, self.obj_normal, (4 positions(theta)-> 나중에 1개로 줄일수도 있음,)
            # 1, 1, 3 , 12 -> 17
            # print(max(self.data.qvel[:self.k]), max(qdot), self.data.qvel[self.k:self.dof])
            self.handle_angle = self.data.qpos[10]
            obs_object = ([self.handle_angle/abs(self.goal_angle)+np.random.normal(0,0.05,1)[0]]
                          +[self.goal_angle/abs(self.goal_angle)]+self.obj_normal+obj_p1+obj_p2+obj_p3+obj_p4)
            self.obs_object = np.array(obs_object).reshape((1, 17))

            distance1 = np.sqrt(
                (end_effector[1][0] - self.obj_normal[0]) ** 2 + (end_effector[1][1] - self.obj_normal[1]) ** 2 + (
                        end_effector[1][2] - self.obj_normal[2]) ** 2)

            distance2 = np.sqrt(
                (end_effector[1][0] - obj_p1[0]) ** 2 + (end_effector[1][1] - obj_p1[1]) ** 2 + (
                        end_effector[1][2] - obj_p1[2]) ** 2)
            distance3 = np.sqrt(
                (end_effector[1][0] - obj_p2[0]) ** 2 + (end_effector[1][1] - obj_p2[1]) ** 2 + (
                        end_effector[1][2] - obj_p2[2]) ** 2)
            distance4 = np.sqrt(
                (end_effector[1][0] - obj_p3[0]) ** 2 + (end_effector[1][1] - obj_p3[1]) ** 2 + (
                        end_effector[1][2] - obj_p3[2]) ** 2)
            distance5 = np.sqrt(
                (end_effector[1][0] - obj_p4[0]) ** 2 + (end_effector[1][1] - obj_p4[1]) ** 2 + (
                        end_effector[1][2] - obj_p4[2]) ** 2)
            self.distance = min([distance1, distance2, distance3, distance4, distance5])
        elif self.obj == "valve":
            self.handle_angle = self.data.qpos[9]
            p1 = self.obj_rotation @ [[0.1* np.cos(self.handle_angle)], [0.1* np.sin(self.handle_angle)], [-0.017]]
            obj_p1 = [0, 0, 0]
            for idx in range(3):
                obj_p1[idx] = p1[idx][0] + self.obj_pos[idx] + np.random.normal(0,0.05,1)[0]


            obs_object = [self.handle_angle / abs(self.goal_angle)+np.random.normal(0,0.05,1)[0]] + [
                self.goal_angle / abs(self.goal_angle)] + self.obj_normal + obj_p1 + obj_p1 + obj_p1 + obj_p1
            self.obs_object = np.array(obs_object).reshape((1, 17))



            distance = np.sqrt(
                (end_effector[1][0] - obj_p1[0]) ** 2 + (end_effector[1][1] - obj_p1[1]) ** 2 + (
                        end_effector[1][2] - obj_p1[2]) ** 2)

            self.distance = distance

        observation = dict(object=self.obs_object,q=self.obs_q,
                           r6d=self.obs_6d, x_pos=self.obs_xyz)

        return observation
    
    def _reward(self,action):
        reward_distance = 0
        reward_goal = 0
        reward_rotation = 0
        reward_time = 0
        reward_grasp = 0
        reward_contact = 0
        reward_bound = 0
        reward_velocity = 0

        # 벨브의 중점으로 부터 일정 이상 가까워 졌을 때
        if self.distance > self.distance_init:
            reward_distance = -1
        else:
            reward_distance = min(pow(((self.distance_init - self.distance)/(self.distance_init-0.1)), 2.73),1)


        # 벨브를 goal angle 만큼 돌렸을 때 -> 끝!
        rotation_angle = self.goal_angle - self.handle_angle
        if self.goal_done: # <- abs(rotation_angle) < 0.1
            reward_goal = 1

        # if abs(rotation_angle) < abs(self.rotation_angle_pre):
        reward_rotation = abs(self.rotation_angle_pre) - abs(rotation_angle)

        if max(abs(self.obs_dq[0])) > 1:
            reward_velocity = 1-max(abs(self.obs_dq[0]))

        # 시간 초과  penalty
        if self.time_done:
            reward_time = -1
        # grasp 한 것에 대한 reward
        if not -1 in self.contact_list:
            reward_grasp = max(0,len(self.grasp_list)-4)  # grasp_list max = 8 : finger parts.
        #충돌 penalty
        if self.contact_done:
            reward_contact = -1
        #q limit 넘어간것 penalty
        if self.bound_done:
            reward_bound = -1

        reward = self.rw_time * reward_time \
                 + self.rw_grasp * reward_grasp \
                 + self.rw_contact * reward_contact \
                 + self.rw_bound * reward_bound \
                 + self.rw_distance * reward_distance \
                 + self.rw_goal * reward_goal \
                 + self.rw_rotation * reward_rotation\
                 + self.rw_velocity * reward_velocity

        # print("reward : ", reward, "acc:",reward_acc," |xyz:",reward_xyz," |rpy:",reward_rpy," |grasp:",reward_grasp)
        self.rotation_angle_pre = rotation_angle
        return reward

    # Overriding
    def _done(self):
        self.contact_list = tools.detect_contact(self.data.contact, self.desired_contact_bid)
        self.grasp_list = tools.detect_grasp(self.data.contact, self.obj, self.desired_contact_finger_bid, self.desired_contact_bid)
        self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)

        self.time_done = self.data.time - self.start_time >= self.episode_time
        self.contact_done = -1 in self.contact_list
        self.bound_done = -1 in self.q_operation_list
        self.goal_done = abs(self.goal_angle - self.handle_angle) < 0.1
        if self.time_done or self.contact_done or self.bound_done or self.goal_done:
            print(self.reward_accum)
            self.reward_accum = 0
            return True
        else:
            return False
    
    # Overriding
    def _construct_action_space(self):
        action_low = np.array([-1.57079, -1.57079, -1.57079, -1.57079, -1.57079, -1.57079, -0.5, -0.5, -0.5, 0])
        action_high = np.array([1.57079, 1.57079, 1.57079, 1.57079, 1.57079, 1.57079, 0.5, 0.5, 0.5, 0.04])
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    
    # Overrding
    def _construct_observation_space(self):
        s = {
            'object': spaces.Box(shape=(1, 17), low=-np.inf, high=np.inf, dtype=np.float32),
            'q': spaces.Box(shape=(self.stack, self.dof), low=-1, high=1, dtype=np.float32),
            'r6d': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'x_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
        }
        return spaces.Dict(s)

    def env_randomization(self):
        obj_list = ["handle", "valve"]
        radius_list = [0.119, 0.1]
        o_margin_list = [[[0], [0.149], [0]], [[0],[0],[-0.017]]]
        o = randint(0, 1)
        obj = obj_list[o]
        o_margin = o_margin_list[o]
        radius = radius_list[o]

        quat_candidate, pos_candidate, nobj = self.read_candidate_json(obj, "candidate_full_action.json")
        bid = mujoco.mj_name2id(self.model, BODY, obj)
        nbid = mujoco.mj_name2id(self.model, BODY, nobj)

        if self.env_rand:
            i = randint(0, 6)
            axis = ['x', 'y', 'z']
            add_quat = R.from_euler(axis[randint(0, 2)], (random() - 0.5) / 5)
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
            # i = 3
            random_quat = quat_candidate[i]
            random_pos = pos_candidate[i]
            self.model.body_quat[bid] = random_quat
            self.model.body_pos[bid] = random_pos
            self.model.body_pos[nbid] += 3
            r = R.from_quat(tools.quat2xyzw(random_quat))
            # random_quat = self.model.body_quat[bid].copy()
            # random_pos =  self.model.body_qpos[bid].copy()
            # r = R.from_quat(tools.quat2xyzw(random_quat))
        mujoco.mj_step(self.model, self.data)

        self.o_margin = o_margin
        self.obj_pos = random_pos
        self.obj_rotation = r.as_matrix()
        self.normal_vector = self.obj_rotation @ self.o_margin
        self.obj_normal = [0,0,0]
        for idx in range(3):
            self.obj_normal[idx] = self.normal_vector[idx][0] + self.obj_pos[idx]

        # current theta, goal theta, self.obj_normal, (4 positions(theta)-> 나중에 1개로 줄일수도 있음,)
        # 1, 1, 3 , 12 -> 17

        return r.as_matrix().tolist(), obj, radius, 0.0