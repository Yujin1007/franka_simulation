import numpy as np
import sys
from numpy.linalg import inv
import mujoco
import gym
from gym import spaces
from random import random, randint, uniform
from scipy.spatial.transform import Rotation as R
from mujoco import viewer
from time import sleep
import tools
import rotations
import torch

BODY = 1
JOINT = 3
GEOM = 5
MOTION_TIME_CONST = 10.
TASK_SPACE_TIME = 3+1+0.5

RL = 2
MANUAL = 1

def BringClassifier(path):
    classifier = torch.load(path)
    classifier.eval()
    return classifier

class fr3_6d_train:
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self) -> None:
        from controller.rpy import controller
        self.k = 7  # for jacobian calculation
        self.dof = 9  # all joints (include gripper joint)
        self.model_path = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/scene_valve.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.controller = controller.CController(self.k)
        self._torque = np.zeros(self.dof, dtype=np.float64)
        self.stack = 5
        self.rendering = False

        self.observation_space = self._construct_observation_space()
        self.action_space = self._construct_action_space()
        ## reward weight
        self.reward_range = None
        self.rw_acc = 1  # np.exp(-sum(abs(action - self.action_pre)))
        self.rw_xyz = 0.1  # np.exp(-2*sum(abs(self.obs_xyz[0] - self.obs_xyzdes[0])))
        self.rw_t = 1  # time done -> reward_time = 1
        self.rw_c = 10  # contact done -> -1
        self.rw_b = 1  # joint boundary done -> -1
        self.rw_gr = 2.0  # 1/-1 grasp
        self.rw_rpy = 0.0  # np.exp(-2*sum(abs(rotations.subtract_euler(self.obs_rpy_des, self.obs_rpy))[0]))
        self.viewer = None
        self.env_rand = False
        self.q_range = self.model.jnt_range[:self.k]
        self.qdot_init = [0,0,0,0,0,0,0,0,0,0,0]
        self.q_init = [0.374, -1.02, 0.245, -1.51, 0.0102, 0.655, 0.3, 0.04, 0.04, 0, 0]
        self.episode_number = -1

        self.classifier_clk = BringClassifier(
            "./classifier/clk/model.pt")
        self.classifier_cclk = BringClassifier(
            "./classifier/cclk/model.pt")
        desired_contact_list = ["finger_contact0", "finger_contact1",
                                "finger_contact2", "finger_contact3", "finger_contact4", "finger_contact5",
                                "finger_contact6", "finger_contact7",
                                "finger_contact8", "finger_contact9", "finger0_contact", "finger1_contact",
                                "handle_contact0", "handle_contact1", "handle_contact2", "handle_contact3",
                                "handle_contact4", "handle_contact5", "handle_contact6", "handle_contact7",
                                "handle_contact8", "handle_contact9", "handle_contact10", "handle_contact11",
                                "handle_contact12", "handle_contact13", "handle_contact14", "handle_contact15",
                                "handle_contact16", "handle_contact17",
                                "finger_contact18", "finger_contact19", "handle_contact20", "handle_contact21",
                                "handle_contact22", "handle_contact23", "valve_contact0", "valve_contact1"]
        desired_contact_list_finger = ["finger_contact1",
                                       "finger_contact2", "finger_contact3", "finger_contact4",
                                       "finger_contact6", "finger_contact7",
                                       "finger_contact8", "finger_contact9", ]
        desired_contact_list_obj = [ "handle_contact0", "handle_contact1",
                                "handle_contact2", "handle_contact3",
                                 "handle_contact5", "handle_contact6",
                                "handle_contact8",  "handle_contact10", "handle_contact11",
                                "handle_contact12", "handle_contact15",
                                "handle_contact16",  "handle_contact21",
                                "handle_contact22", "handle_contact23", "valve_contact0"]

        robot_contact_list = ["link0_contact", "link1_contact", "link2_contact", "link3_contact", \
                                   "link4_contact", "link5_contact0", "link5_contact1", "link5_contact2", \
                                   "link6_contact", "link7_contact", "hand_contact", "finger0_contact", \
                                   "finger1_contact", "finger_contact0", "finger_contact1", "finger_contact2",\
                                   "finger_contact3", "finger_contact4", "finger_contact5", "finger_contact6",\
                                   "finger_contact7", "finger_contact8", "finger_contact9"]

        object_contact_list = ["handle_contact0", "handle_contact1", "handle_contact2", "handle_contact3", \
                                    "handle_contact4", "handle_contact5", "handle_contact6", "handle_contact7", \
                                    "handle_contact8", "handle_contact9", "valve_contact0",\
                                    "valve_contact0", "valve_contact1", "valve_contact2", "valve_contact3",\
                                    "valve_contact4", "valve_contact5", "valve_contact6", "valve_contact7",\
                                    "valve_contact8", "valve_contact9", "valve_contact10", "valve_contact11",\
                                    "valve_contact12", "valve_contact13", "valve_contact0"]

        self.desired_contact_bid = tools.name2id(self.model, GEOM, desired_contact_list)
        self.desired_contact_finger_bid = tools.name2id(self.model, GEOM, desired_contact_list_finger)
        self.desired_contact_obj_bid = tools.name2id(self.model, GEOM, desired_contact_list_obj)
        self.robot_contact_bid = tools.name2id(self.model, GEOM, robot_contact_list)
        self.object_contact_bid = tools.name2id(self.model, GEOM, object_contact_list)

        self.ADR_threshold = 20
        self.ADR_cnt = 0
        self.ADR_object = 1

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
            r, obj, radius, init_angle = self.env_randomization() #self.obs_object initialize

            self.init_angle = init_angle

            if self.direction == "clk":
                self.goal_angle = init_angle - 2 * np.pi
            elif self.direction == "cclk":
                self.goal_angle = init_angle + 2 * np.pi
            else:
                self.goal_angle = init_angle + 2 * np.pi * (-1) ** self.episode_number  # 2*np.pi
            self.episode_time = abs( MOTION_TIME_CONST * abs(self.goal_angle-self.init_angle) * radius) + TASK_SPACE_TIME
            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
            self.controller.randomize_env(r, obj, self.init_angle, self.goal_angle, RL)

            self.controller.control_mujoco()
            self.start_time = self.data.time


            self.time_done = False
            self.contact_done = False
            self.bound_done = False
            self.action_pre  = np.zeros(6)
            self.drpy_pre  = np.zeros(3)



            self.obs_q = np.zeros([self.stack, self.k])
            self.obs_6d = np.zeros([self.stack, 6])
            self.obs_rpy = np.zeros([self.stack,6])
            self.obs_rpy_des = np.zeros([self.stack,6])
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


    def run(self, iteration) -> None:
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



    def step(self, action):
        # print("action  : ", action)
        drpy = tools.orientation_6d_to_euler(action)
        # print(drpy)
        # drpy = action
        while self.control_mode != 4:
            self.control_mode = self.controller.control_mode()
            # self.control_mode = 0
            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
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
                # sleep(0.002)
        for j in range(100): # 1000hz -> 10hz

            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
            if j<10:
                drpy_tmp = (drpy - self.drpy_pre)/10 * j + self.drpy_pre
                self.controller.put_action(drpy_tmp)
            else:
                self.controller.put_action(drpy)
            # self.controller.put_action(action)

            # calculate x_des_hand.head(3) and generate control torque
            self.controller.control_mujoco()

            # take action

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
        rpy_des = self.controller.desired_rpy()
        obs = self._observation(ee,rpy_des)
        done = self._done()
        reward = self._reward(action)
        info = self._info()

        self.drpy_pre = drpy
        self.action_pre = action

        return obs, reward, done, info

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
        self.obs_rpy[0] = tools.orientation_euler_to_6d(rpy)
        self.obs_rpy_des[0] = tools.orientation_euler_to_6d(rpy_des)
        self.obs_manipulability[0] = tools.calc_manipulability(jacobian)
        # print(np.round(self.obs_manipulability[0]))
        # print(max(self.obs_manipulability[0]), min(self.obs_manipulability[0]),"\n")
        observation = dict(object=self.obs_object,q=self.obs_q,rpy=self.obs_rpy, rpy_des=self.obs_rpy_des, x_plan=self.obs_xyzdes, x_pos=self.obs_xyz)
        self.save_frame_data(end_effector)
        return observation
    def _reward(self,action):
        reward_acc = np.exp(-sum(abs(action - self.action_pre))) #max = -12 * const
        reward_xyz = np.exp(-2*sum(abs(self.obs_xyz[0] - self.obs_xyzdes[0])))
        reward_rpy = np.exp(-2 * sum(abs(rotations.subtract_euler(tools.orientation_6d_to_euler(self.obs_rpy_des[0]),
                                                                  tools.orientation_6d_to_euler(self.obs_rpy[0])))))

        # print(reward_rpy)
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
                + self.rw_rpy*reward_rpy
        # print("reward : ", reward, "acc:",reward_acc," |xyz:",reward_xyz," |rpy:",reward_rpy," |grasp:",reward_grasp)
        return reward


    def _done(self):

        self.contact_list = tools.detect_contact(self.data.contact, self.desired_contact_bid)
        self.grasp_list = tools.detect_grasp(self.data.contact, self.obj, self.desired_contact_finger_bid, self.desired_contact_obj_bid)
        self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)

        self.time_done = self.data.time - self.start_time >= self.episode_time
        self.contact_done = -1 in self.contact_list
        self.bound_done = -1 in self.q_operation_list
        if self.time_done or self.contact_done or self.bound_done :
            # print("contact :", self.contact_done, "  //joint :", self.bound_done, "  //time :", self.time_done,
            #       "  //torque :", self.torque_done)
            # print("epispde time : ",self.episode_time, "time:",self.data.time-self.start_time)

            return True
        else:
            return False
    def _info(self):
        info = {
            "collision": self.contact_done,
            "bound" : self.bound_done,
        }
        return info
    def _construct_action_space(self):
        action_space = 6
        action_low = -1*np.ones(action_space)
        action_high = 1* np.ones(action_space)
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def _construct_observation_space(self):

        s = {
            'object': spaces.Box(shape=(1, 14), low=-np.inf, high=np.inf, dtype=np.float32),
            'q': spaces.Box(shape=(self.stack, self.k), low=-1, high=1, dtype=np.float32),
            'rpy': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'rpy_des': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'x_plan': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
            'x_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
        }

        return spaces.Dict(s)
    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        else:
            self.viewer.sync()
            # sleep(0.002)

    def env_randomization(self):

        obj_list = ["handle", "valve"]
        radius_list = [0.119, 0.1]
        o = randint(0,1)
        obj = obj_list[o]
        radius = radius_list[o]
        handle_quat_candidate = [[0.25192415, -0.64412663, 0.57897236, 0.4317709],
                                 [-0.49077636, 0.42062713, -0.75930974, 0.07523369],
                                 [0.474576307745582, -0.089013785474907, 0.275616460318178, 0.831197594392378],
                                 [0., -0.707, 0.707, 0.],
                                 [-0.46086475, -0.63305975, 0.39180338, 0.48304156],
                                 [-0.07865809, -0.89033475, 0.16254433, -0.41796684],
                                 [0.70738827, 0., 0., 0.70682518]]
        handle_pos_candidate = [[0.52, 0, 0.8],
                                [0.38, 0, 0.9],
                                [0.326, 0.232, 0.559+0.35],
                                [0.55, 0., 0.75],
                                [0.4, 0.3, 0.5],
                                [0.35, 0.45, 0.9],
                                [0.48, 0, 0.9]]
        valve_quat_candidate = [[0., 1., 0., 0.],
                                [-0.707, 0.707, 0., 0.],
                                [0., -0.707, 0.707, 0.],
                                [0., -0.707, 0.707, 0.],
                                [0., 0.707, - 0., 0.707],
                                [-0.707, 0.707, 0., 0.],
                                [0., 1., 0., 0.]]
        valve_pos_candidate = [[0.38, 0., 0.45],
                               [-0.2, 0.5, 0.6],
                               [0.28, 0., 0.7],
                               [0.38, 0., 0.5],
                               [0.48, 0., 0.55],
                               [0.3, 0.3, 0.6],
                               [0.3, 0.3, 0.6]]
        if obj == "handle":
            nobj = "valve"
            quat_candidate = handle_quat_candidate
            pos_candidate = handle_pos_candidate
        elif obj == "valve":
            nobj = "handle"
            quat_candidate = valve_quat_candidate
            pos_candidate = valve_pos_candidate

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
                result = result + add_idx

        elif obj == "valve":
            obj_id = [0, 1, 0]
            result = 0
            self.o_margin = [[0], [0], [-0.017]]
            self.T_vv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            direction = 0

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

    def save_frame_data(self, ee):
        r = R.from_euler('xyz', ee[1][3:6], degrees=False)
        rpyfromvalve_rot = r.inv() * R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)
        ee_align = R.from_euler('z', 45, degrees=True)
        rpyfromvalve = (ee_align * rpyfromvalve_rot).as_matrix()

        xyzfromvalve_rot = (R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)).as_matrix()
        xyzfromvalve_rot = np.concatenate([xyzfromvalve_rot, [[0, 0, 0]]], axis=0)
        xyzfromvalve_rot = np.concatenate(
            [xyzfromvalve_rot, [[self.obj_normal[0]], [self.obj_normal[1]], [self.obj_normal[2]], [1]]], axis=1)

        xyzfromvalve = inv(xyzfromvalve_rot) @ np.array([[ee[1][0]], [ee[1][1]], [ee[1][2]], [1]])

        if len(self.rpyfromvalve_data) == 0:
            self.rpyfromvalve_data = rpyfromvalve.reshape(1, 3, 3)
            self.xyzfromvalve_data = xyzfromvalve[0:3].reshape(1, 3)
            self.gripper_data = [ee[2][0][0]]
        else:
            self.rpyfromvalve_data = np.concatenate([self.rpyfromvalve_data, [rpyfromvalve]], axis=0)
            self.xyzfromvalve_data = np.concatenate([self.xyzfromvalve_data, [xyzfromvalve[0:3].reshape(3)]], axis=0)
            self.gripper_data = np.concatenate([self.gripper_data, [ee[2][0][0]]], axis=0)


    def read_file(self):
        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dr_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dr = list(map(float, f_list))
        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dp_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dp = list(map(float, f_list))

        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dy_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dy = list(map(float, f_list))

class fr3_6d_test:
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self) -> None:
        from controller.rpy import controller
        self.k = 7  # for jacobian calculation
        self.dof = 9  # all joints (include gripper joint)
        self.model_path = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/scene_valve.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.controller = controller.CController(self.k)
        self._torque = np.zeros(self.dof, dtype=np.float64)
        self.stack = 5
        self.rendering = False

        self.observation_space = self._construct_observation_space()
        self.action_space = self._construct_action_space()
        ## reward weight
        self.reward_range = None
        self.rw_acc = 1  # np.exp(-sum(abs(action - self.action_pre)))
        self.rw_xyz = 0.1  # np.exp(-2*sum(abs(self.obs_xyz[0] - self.obs_xyzdes[0])))
        self.rw_t = 1  # time done -> reward_time = 1
        self.rw_c = 10  # contact done -> -1
        self.rw_b = 1  # joint boundary done -> -1
        self.rw_gr = 2.0  # 1/-1 grasp
        self.rw_rpy = 0.0  # np.exp(-2*sum(abs(rotations.subtract_euler(self.obs_rpy_des, self.obs_rpy))[0]))
        self.viewer = None
        self.env_rand = False
        self.q_range = self.model.jnt_range[:self.k]
        self.qdot_init = [0,0,0,0,0,0,0,0,0,0,0]
        self.q_init = [0.374, -1.02, 0.245, -1.51, 0.0102, 0.655, 0.3, 0.04, 0.04, 0, 0]
        self.episode_number = -1

        self.classifier_clk = BringClassifier(
            "./classifier/clk/model.pt")
        self.classifier_cclk = BringClassifier(
            "./classifier/cclk/model.pt")

        desired_contact_list = ["finger_contact0", "finger_contact1",
                                "finger_contact2", "finger_contact3", "finger_contact4", "finger_contact5",
                                "finger_contact6", "finger_contact7",
                                "finger_contact8", "finger_contact9", "finger0_contact", "finger1_contact",
                                "handle_contact0", "handle_contact1", "handle_contact2", "handle_contact3",
                                "handle_contact4", "handle_contact5", "handle_contact6", "handle_contact7",
                                "handle_contact8", "handle_contact9", "handle_contact10", "handle_contact11",
                                "handle_contact12", "handle_contact13", "handle_contact14", "handle_contact15",
                                "handle_contact16", "handle_contact17",
                                "finger_contact18", "finger_contact19", "handle_contact20", "handle_contact21",
                                "handle_contact22", "handle_contact23", "valve_contact0", "valve_contact1"]
        desired_contact_list_finger = ["finger_contact1",
                                       "finger_contact2", "finger_contact3", "finger_contact4",
                                       "finger_contact6", "finger_contact7",
                                       "finger_contact8", "finger_contact9", ]
        desired_contact_list_obj = [ "handle_contact0", "handle_contact1",
                                "handle_contact2", "handle_contact3",
                                 "handle_contact5", "handle_contact6",
                                "handle_contact8",  "handle_contact10", "handle_contact11",
                                "handle_contact12", "handle_contact15",
                                "handle_contact16",  "handle_contact21",
                                "handle_contact22", "handle_contact23", "valve_contact0"]

        robot_contact_list = ["link0_contact", "link1_contact", "link2_contact", "link3_contact", \
                                   "link4_contact", "link5_contact0", "link5_contact1", "link5_contact2", \
                                   "link6_contact", "link7_contact", "hand_contact", "finger0_contact", \
                                   "finger1_contact", "finger_contact0", "finger_contact1", "finger_contact2",\
                                   "finger_contact3", "finger_contact4", "finger_contact5", "finger_contact6",\
                                   "finger_contact7", "finger_contact8", "finger_contact9"]

        object_contact_list = ["handle_contact0", "handle_contact1", "handle_contact2", "handle_contact3", \
                                    "handle_contact4", "handle_contact5", "handle_contact6", "handle_contact7", \
                                    "handle_contact8", "handle_contact9", "valve_contact0",\
                                    "valve_contact0", "valve_contact1", "valve_contact2", "valve_contact3",\
                                    "valve_contact4", "valve_contact5", "valve_contact6", "valve_contact7",\
                                    "valve_contact8", "valve_contact9", "valve_contact10", "valve_contact11",\
                                    "valve_contact12", "valve_contact13", "valve_contact0"]

        self.desired_contact_bid = tools.name2id(self.model, GEOM, desired_contact_list)
        self.desired_contact_finger_bid = tools.name2id(self.model, GEOM, desired_contact_list_finger)
        self.desired_contact_obj_bid = tools.name2id(self.model, GEOM, desired_contact_list_obj)
        self.robot_contact_bid = tools.name2id(self.model, GEOM, robot_contact_list)
        self.object_contact_bid = tools.name2id(self.model, GEOM, object_contact_list)

        self.ADR_threshold = 20
        self.ADR_cnt = 0
        self.ADR_object = 1

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
                self.controller.randomize_env(self.r, self.obj, self.init_angle, self.goal_angle, RL)
            else:
                self.controller.randomize_env(self.r, self.obj, self.init_angle, self.goal_angle, MANUAL)
            self.controller.control_mujoco()
            self.start_time = self.data.time


            self.time_done = False
            self.contact_done = False
            self.bound_done = False
            self.action_pre  = np.zeros(6)
            self.drpy_pre  = np.zeros(3)



            self.obs_q = np.zeros([self.stack, self.k])
            self.obs_6d = np.zeros([self.stack, 6])
            self.obs_rpy = np.zeros([self.stack,6])
            self.obs_rpy_des = np.zeros([self.stack,6])
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


    def run(self, iteration) -> None:
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



    def step(self, action):
        # print("action  : ", action)
        drpy = tools.orientation_6d_to_euler(action)
        # print(drpy)
        # drpy = action
        while self.control_mode != 4:
            self.control_mode = self.controller.control_mode()
            # self.control_mode = 0
            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
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
                # sleep(0.002)
        for j in range(100): # 1000hz -> 10hz

            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
            if j<10:
                drpy_tmp = (drpy - self.drpy_pre)/10 * j + self.drpy_pre
                self.controller.put_action(drpy_tmp)
            else:
                self.controller.put_action(drpy)
            # self.controller.put_action(action)

            # calculate x_des_hand.head(3) and generate control torque
            self.controller.control_mujoco()

            # take action

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
        rpy_des = self.controller.desired_rpy()
        obs = self._observation(ee,rpy_des)
        done = self._done()
        reward = self._reward(action)
        info = self._info()

        self.drpy_pre = drpy
        self.action_pre = action

        return obs, reward, done, info

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
        self.obs_rpy[0] = tools.orientation_euler_to_6d(rpy)
        self.obs_rpy_des[0] = tools.orientation_euler_to_6d(rpy_des)
        self.obs_manipulability[0] = tools.calc_manipulability(jacobian)
        # print(np.round(self.obs_manipulability[0]))
        # print(max(self.obs_manipulability[0]), min(self.obs_manipulability[0]),"\n")
        observation = dict(object=self.obs_object,q=self.obs_q,rpy=self.obs_rpy, rpy_des=self.obs_rpy_des, x_plan=self.obs_xyzdes, x_pos=self.obs_xyz)
        self.save_frame_data(end_effector)
        return observation
    def _reward(self,action):
        reward_acc = np.exp(-sum(abs(action - self.action_pre))) #max = -12 * const
        reward_xyz = np.exp(-2*sum(abs(self.obs_xyz[0] - self.obs_xyzdes[0])))
        reward_rpy = np.exp(-2 * sum(abs(rotations.subtract_euler(tools.orientation_6d_to_euler(self.obs_rpy_des[0]),
                                                                  tools.orientation_6d_to_euler(self.obs_rpy[0])))))

        # print(reward_rpy)
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
                + self.rw_rpy*reward_rpy
        # print("reward : ", reward, "acc:",reward_acc," |xyz:",reward_xyz," |rpy:",reward_rpy," |grasp:",reward_grasp)
        return reward


    def _done(self):

        self.contact_list = tools.detect_contact(self.data.contact, self.desired_contact_bid)
        self.grasp_list = tools.detect_grasp(self.data.contact, self.obj, self.desired_contact_finger_bid, self.desired_contact_obj_bid)
        self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)

        self.time_done = self.data.time - self.start_time >= self.episode_time
        self.contact_done = -1 in self.contact_list
        self.bound_done = -1 in self.q_operation_list
        if self.time_done or self.contact_done or self.bound_done :
            # print("contact :", self.contact_done, "  //joint :", self.bound_done, "  //time :", self.time_done,
            #       "  //torque :", self.torque_done)
            # print("epispde time : ",self.episode_time, "time:",self.data.time-self.start_time)

            return True
        else:
            return False
    def _info(self):
        info = {
            "collision": self.contact_done,
            "bound" : self.bound_done,
        }
        return info
    def _construct_action_space(self):
        action_space = 6
        action_low = -1*np.ones(action_space)
        action_high = 1* np.ones(action_space)
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def _construct_observation_space(self):

        s = {
            'object': spaces.Box(shape=(1, 14), low=-np.inf, high=np.inf, dtype=np.float32),
            'q': spaces.Box(shape=(self.stack, self.k), low=-1, high=1, dtype=np.float32),
            'rpy': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'rpy_des': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'x_plan': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
            'x_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
        }

        return spaces.Dict(s)
    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        else:
            self.viewer.sync()
            # sleep(0.002)

    def env_randomization(self):

        obj_list = ["handle", "valve"]
        radius_list = [0.119, 0.1]
        o = randint(0,1)
        o = 0
        obj = obj_list[o]
        radius = radius_list[o]
        handle_quat_candidate = [[0.25192415, -0.64412663, 0.57897236, 0.4317709],
                                 [-0.49077636, 0.42062713, -0.75930974, 0.07523369],
                                 [0.474576307745582, -0.089013785474907, 0.275616460318178, 0.831197594392378],
                                 [0., -0.707, 0.707, 0.],
                                 [-0.46086475, -0.63305975, 0.39180338, 0.48304156],
                                 [-0.07865809, -0.89033475, 0.16254433, -0.41796684],
                                 [0.70738827, 0., 0., 0.70682518]]
        handle_pos_candidate = [[0.52, 0, 0.8],
                                [0.38, 0, 0.9],
                                [0.326, 0.232, 0.559+0.35],
                                [0.55, 0., 0.75],
                                [0.4, 0.3, 0.5],
                                [0.35, 0.45, 0.9],
                                [0.48, 0, 0.9]]
        valve_quat_candidate = [[0., 1., 0., 0.],
                                [-0.707, 0.707, 0., 0.],
                                [0., -0.707, 0.707, 0.],
                                [0., -0.707, 0.707, 0.],
                                [0., 0.707, - 0., 0.707],
                                [-0.707, 0.707, 0., 0.],
                                [0., 1., 0., 0.]]
        valve_pos_candidate = [[0.38, 0., 0.45],
                               [-0.2, 0.5, 0.6],
                               [0.28, 0., 0.7],
                               [0.38, 0., 0.5],
                               [0.48, 0., 0.55],
                               [0.3, 0.3, 0.6],
                               [0.3, 0.3, 0.6]]
        if obj == "handle":
            nobj = "valve"
            quat_candidate = handle_quat_candidate
            pos_candidate = handle_pos_candidate
        elif obj == "valve":
            nobj = "handle"
            quat_candidate = valve_quat_candidate
            pos_candidate = valve_pos_candidate

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

    def save_frame_data(self, ee):
        r = R.from_euler('xyz', ee[1][3:6], degrees=False)
        rpyfromvalve_rot = r.inv() * R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)
        ee_align = R.from_euler('z', 45, degrees=True)
        rpyfromvalve = (ee_align * rpyfromvalve_rot).as_matrix()

        xyzfromvalve_rot = (R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)).as_matrix()
        xyzfromvalve_rot = np.concatenate([xyzfromvalve_rot, [[0, 0, 0]]], axis=0)
        xyzfromvalve_rot = np.concatenate(
            [xyzfromvalve_rot, [[self.obj_normal[0]], [self.obj_normal[1]], [self.obj_normal[2]], [1]]], axis=1)

        xyzfromvalve = inv(xyzfromvalve_rot) @ np.array([[ee[1][0]], [ee[1][1]], [ee[1][2]], [1]])

        if len(self.rpyfromvalve_data) == 0:
            self.rpyfromvalve_data = rpyfromvalve.reshape(1, 3, 3)
            self.xyzfromvalve_data = xyzfromvalve[0:3].reshape(1, 3)
            self.gripper_data = [ee[2][0][0]]
        else:
            self.rpyfromvalve_data = np.concatenate([self.rpyfromvalve_data, [rpyfromvalve]], axis=0)
            self.xyzfromvalve_data = np.concatenate([self.xyzfromvalve_data, [xyzfromvalve[0:3].reshape(3)]], axis=0)
            self.gripper_data = np.concatenate([self.gripper_data, [ee[2][0][0]]], axis=0)


    def read_file(self):
        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dr_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dr = list(map(float, f_list))
        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dp_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dp = list(map(float, f_list))

        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dy_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dy = list(map(float, f_list))

class fr3_3d_test:
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self) -> None:
        from controller.rpy import controller
        self.k = 7  # for jacobian calculation
        self.dof = 9  # all joints (include gripper joint)
        self.model_path = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/scene_valve.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.controller = controller.CController(self.k)
        self._torque = np.zeros(self.dof, dtype=np.float64)
        self.stack = 5
        self.rendering = False

        self.observation_space = self._construct_observation_space()
        self.action_space = self._construct_action_space()
        ## reward weight
        self.reward_range = None
        self.rw_acc = 1  # np.exp(-sum(abs(action - self.action_pre)))
        self.rw_xyz = 0.1  # np.exp(-2*sum(abs(self.obs_xyz[0] - self.obs_xyzdes[0])))
        self.rw_t = 1  # time done -> reward_time = 1
        self.rw_c = 10  # contact done -> -1
        self.rw_b = 1  # joint boundary done -> -1
        self.rw_gr = 2.0  # 1/-1 grasp
        self.rw_rpy = 0.0  # np.exp(-2*sum(abs(rotations.subtract_euler(self.obs_rpy_des, self.obs_rpy))[0]))
        self.viewer = None
        self.env_rand = False
        self.q_range = self.model.jnt_range[:self.k]
        self.qdot_init = [0,0,0,0,0,0,0,0,0,0,0]
        self.q_init = [0.374, -1.02, 0.245, -1.51, 0.0102, 0.655, 0.3, 0.04, 0.04, 0, 0]
        self.episode_number = -1

        self.classifier_clk = BringClassifier(
            "./classifier/clk/model.pt")
        self.classifier_cclk = BringClassifier(
            "./classifier/cclk/model.pt")
        desired_contact_list = ["finger_contact0", "finger_contact1",
                                "finger_contact2", "finger_contact3", "finger_contact4", "finger_contact5",
                                "finger_contact6", "finger_contact7",
                                "finger_contact8", "finger_contact9", "finger0_contact", "finger1_contact",
                                "handle_contact0", "handle_contact1", "handle_contact2", "handle_contact3",
                                "handle_contact4", "handle_contact5", "handle_contact6", "handle_contact7",
                                "handle_contact8", "handle_contact9", "handle_contact10", "handle_contact11",
                                "handle_contact12", "handle_contact13", "handle_contact14", "handle_contact15",
                                "handle_contact16", "handle_contact17",
                                "finger_contact18", "finger_contact19", "handle_contact20", "handle_contact21",
                                "handle_contact22", "handle_contact23", "valve_contact0", "valve_contact1"]
        desired_contact_list_finger = ["finger_contact1",
                                       "finger_contact2", "finger_contact3", "finger_contact4",
                                       "finger_contact6", "finger_contact7",
                                       "finger_contact8", "finger_contact9", ]
        desired_contact_list_obj = [ "handle_contact0", "handle_contact1",
                                "handle_contact2", "handle_contact3",
                                 "handle_contact5", "handle_contact6",
                                "handle_contact8",  "handle_contact10", "handle_contact11",
                                "handle_contact12", "handle_contact15",
                                "handle_contact16",  "handle_contact21",
                                "handle_contact22", "handle_contact23", "valve_contact0"]

        robot_contact_list = ["link0_contact", "link1_contact", "link2_contact", "link3_contact", \
                                   "link4_contact", "link5_contact0", "link5_contact1", "link5_contact2", \
                                   "link6_contact", "link7_contact", "hand_contact", "finger0_contact", \
                                   "finger1_contact", "finger_contact0", "finger_contact1", "finger_contact2",\
                                   "finger_contact3", "finger_contact4", "finger_contact5", "finger_contact6",\
                                   "finger_contact7", "finger_contact8", "finger_contact9"]

        object_contact_list = ["handle_contact0", "handle_contact1", "handle_contact2", "handle_contact3", \
                                    "handle_contact4", "handle_contact5", "handle_contact6", "handle_contact7", \
                                    "handle_contact8", "handle_contact9", "valve_contact0",\
                                    "valve_contact0", "valve_contact1", "valve_contact2", "valve_contact3",\
                                    "valve_contact4", "valve_contact5", "valve_contact6", "valve_contact7",\
                                    "valve_contact8", "valve_contact9", "valve_contact10", "valve_contact11",\
                                    "valve_contact12", "valve_contact13", "valve_contact0"]

        self.desired_contact_bid = tools.name2id(self.model, GEOM, desired_contact_list)
        self.desired_contact_finger_bid = tools.name2id(self.model, GEOM, desired_contact_list_finger)
        self.desired_contact_obj_bid = tools.name2id(self.model, GEOM, desired_contact_list_obj)
        self.robot_contact_bid = tools.name2id(self.model, GEOM, robot_contact_list)
        self.object_contact_bid = tools.name2id(self.model, GEOM, object_contact_list)

        self.ADR_threshold = 20
        self.ADR_cnt = 0
        self.ADR_object = 1

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
                self.controller.randomize_env(self.r, self.obj, self.init_angle, self.goal_angle, RL)
            else:
                self.controller.randomize_env(self.r, self.obj, self.init_angle, self.goal_angle, MANUAL)
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


    def run(self, iteration) -> None:
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



    def step(self, action):
        # print("action  : ", action)
        # drpy = tools.orientation_6d_to_euler(action)
        # print(drpy)
        drpy = action
        while self.control_mode != 4:
            self.control_mode = self.controller.control_mode()
            # self.control_mode = 0
            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
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
                # sleep(0.002)
        for j in range(100): # 1000hz -> 10hz

            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66, ))
            if j<10:
                drpy_tmp = (drpy - self.drpy_pre)/10 * j + self.drpy_pre
                self.controller.put_action(drpy_tmp)
            else:
                self.controller.put_action(drpy)
            # self.controller.put_action(action)

            # calculate x_des_hand.head(3) and generate control torque
            self.controller.control_mujoco()

            # take action

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
        rpy_des = self.controller.desired_rpy()
        obs = self._observation(ee,rpy_des)
        done = self._done()
        reward = self._reward(action)
        info = self._info()

        self.drpy_pre = drpy
        self.action_pre = action

        return obs, reward, done, info

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
        self.obs_rpy[0] = rpy
        self.obs_rpy_des[0] = rpy_des
        self.obs_manipulability[0] = tools.calc_manipulability(jacobian)
        # print(np.round(self.obs_manipulability[0]))
        # print(max(self.obs_manipulability[0]), min(self.obs_manipulability[0]),"\n")
        observation = dict(object=self.obs_object,q=self.obs_q,rpy=self.obs_rpy, rpy_des=self.obs_rpy_des, x_plan=self.obs_xyzdes, x_pos=self.obs_xyz)
        self.save_frame_data(end_effector)
        return observation
    def _reward(self,action):
        reward_acc = np.exp(-sum(abs(action - self.action_pre))) #max = -12 * const
        reward_xyz = np.exp(-2*sum(abs(self.obs_xyz[0] - self.obs_xyzdes[0])))
        reward_rpy = np.exp(-2 * sum(abs(rotations.subtract_euler(self.obs_rpy_des[0], self.obs_rpy[0]))))

        # print(reward_rpy)
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
                + self.rw_rpy*reward_rpy
        # print("reward : ", reward, "acc:",reward_acc," |xyz:",reward_xyz," |rpy:",reward_rpy," |grasp:",reward_grasp)
        return reward


    def _done(self):

        self.contact_list = tools.detect_contact(self.data.contact, self.desired_contact_bid)
        self.grasp_list = tools.detect_grasp(self.data.contact, self.obj, self.desired_contact_finger_bid, self.desired_contact_obj_bid)
        self.q_operation_list = tools.detect_q_operation(self.data.qpos, self.q_range)

        self.time_done = self.data.time - self.start_time >= self.episode_time
        self.contact_done = -1 in self.contact_list
        self.bound_done = -1 in self.q_operation_list
        if self.time_done or self.contact_done or self.bound_done :
            # print("contact :", self.contact_done, "  //joint :", self.bound_done, "  //time :", self.time_done,
            #       "  //torque :", self.torque_done)
            # print("epispde time : ",self.episode_time, "time:",self.data.time-self.start_time)

            return True
        else:
            return False
    def _info(self):
        info = {
            "collision": self.contact_done,
            "bound" : self.bound_done,
        }
        return info
    def _construct_action_space(self):
        action_space = 6
        action_low = -1*np.ones(action_space)
        action_high = 1* np.ones(action_space)
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)

    def _construct_observation_space(self):

        s = {
            'object': spaces.Box(shape=(1, 14), low=-np.inf, high=np.inf, dtype=np.float32),
            'q': spaces.Box(shape=(self.stack, self.k), low=-1, high=1, dtype=np.float32),
            'rpy': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'rpy_des': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'x_plan': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
            'x_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
        }

        return spaces.Dict(s)
    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        else:
            self.viewer.sync()
            # sleep(0.002)

    def env_randomization(self):

        obj_list = ["handle", "valve"]
        radius_list = [0.119, 0.1]
        o = randint(0,1)
        o=0
        obj = obj_list[o]
        radius = radius_list[o]
        handle_quat_candidate = [[0.25192415, -0.64412663, 0.57897236, 0.4317709],
                                 [-0.49077636, 0.42062713, -0.75930974, 0.07523369],
                                 [0.474576307745582, -0.089013785474907, 0.275616460318178, 0.831197594392378],
                                 [0., -0.707, 0.707, 0.],
                                 [-0.46086475, -0.63305975, 0.39180338, 0.48304156],
                                 [-0.07865809, -0.89033475, 0.16254433, -0.41796684],
                                 [0.70738827, 0., 0., 0.70682518]]
        handle_pos_candidate = [[0.52, 0, 0.8],
                                [0.38, 0, 0.9],
                                [0.326, 0.232, 0.559+0.35],
                                [0.55, 0., 0.75],
                                [0.4, 0.3, 0.5],
                                [0.35, 0.45, 0.9],
                                [0.48, 0, 0.9]]
        valve_quat_candidate = [[0., 1., 0., 0.],
                                [-0.707, 0.707, 0., 0.],
                                [0., -0.707, 0.707, 0.],
                                [0., -0.707, 0.707, 0.],
                                [0., 0.707, - 0., 0.707],
                                [-0.707, 0.707, 0., 0.],
                                [0., 1., 0., 0.]]
        valve_pos_candidate = [[0.38, 0., 0.45],
                               [-0.2, 0.5, 0.6],
                               [0.28, 0., 0.7],
                               [0.38, 0., 0.5],
                               [0.48, 0., 0.55],
                               [0.3, 0.3, 0.6],
                               [0.3, 0.3, 0.6]]
        if obj == "handle":
            nobj = "valve"
            quat_candidate = handle_quat_candidate
            pos_candidate = handle_pos_candidate
        elif obj == "valve":
            nobj = "handle"
            quat_candidate = valve_quat_candidate
            pos_candidate = valve_pos_candidate

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

    def save_frame_data(self, ee):
        r = R.from_euler('xyz', ee[1][3:6], degrees=False)
        rpyfromvalve_rot = r.inv() * R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)
        ee_align = R.from_euler('z', 45, degrees=True)
        rpyfromvalve = (ee_align * rpyfromvalve_rot).as_matrix()

        xyzfromvalve_rot = (R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)).as_matrix()
        xyzfromvalve_rot = np.concatenate([xyzfromvalve_rot, [[0, 0, 0]]], axis=0)
        xyzfromvalve_rot = np.concatenate(
            [xyzfromvalve_rot, [[self.obj_normal[0]], [self.obj_normal[1]], [self.obj_normal[2]], [1]]], axis=1)

        xyzfromvalve = inv(xyzfromvalve_rot) @ np.array([[ee[1][0]], [ee[1][1]], [ee[1][2]], [1]])

        if len(self.rpyfromvalve_data) == 0:
            self.rpyfromvalve_data = rpyfromvalve.reshape(1, 3, 3)
            self.xyzfromvalve_data = xyzfromvalve[0:3].reshape(1, 3)
            self.gripper_data = [ee[2][0][0]]
        else:
            self.rpyfromvalve_data = np.concatenate([self.rpyfromvalve_data, [rpyfromvalve]], axis=0)
            self.xyzfromvalve_data = np.concatenate([self.xyzfromvalve_data, [xyzfromvalve[0:3].reshape(3)]], axis=0)
            self.gripper_data = np.concatenate([self.gripper_data, [ee[2][0][0]]], axis=0)


    def read_file(self):
        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dr_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dr = list(map(float, f_list))
        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dp_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dp = list(map(float, f_list))

        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dy_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dy = list(map(float, f_list))

class fr3_full_action:
    metadata = {"render_modes": ["human"], "render_fps": 30}


    def __init__(self) -> None:
        from controller.full_action import controller
        self.k = 7  # for jacobian calculation
        self.dof = 9  # all joints (include gripper joint)
        self.model_path = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/scene_valve.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.controller = controller.CController(self.k)
        self._torque = np.zeros(self.dof, dtype=np.float64)
        self.stack = 5
        self.rendering = False
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

        desired_contact_list = ["finger_contact0", "finger_contact1",
                                "finger_contact2", "finger_contact3", "finger_contact4", "finger_contact5",
                                "finger_contact6", "finger_contact7","finger0_contact", "finger1_contact",
                                "finger_contact8", "finger_contact9", "hand_contact",
                                "handle_contact0", "handle_contact1", "handle_contact2", "handle_contact3",
                                "handle_contact4", "handle_contact5", "handle_contact6", "handle_contact7",
                                "handle_contact8", "handle_contact9", "handle_contact10", "handle_contact11",
                                "handle_contact12", "handle_contact13", "handle_contact14", "handle_contact15",
                                "handle_contact16", "handle_contact17",
                                "finger_contact18", "finger_contact19", "handle_contact20", "handle_contact21",
                                "handle_contact22", "handle_contact23", "valve_contact0", "valve_contact1"]
        desired_contact_list_finger = ["finger_contact1",
                                       "finger_contact2", "finger_contact3", "finger_contact4",
                                       "finger_contact6", "finger_contact7",
                                       "finger_contact8", "finger_contact9", ]
        desired_contact_list_obj = ["handle_contact0", "handle_contact1",
                                    "handle_contact2", "handle_contact3",
                                    "handle_contact5", "handle_contact6",
                                    "handle_contact8", "handle_contact10", "handle_contact11",
                                    "handle_contact12", "handle_contact15",
                                    "handle_contact16", "handle_contact21",
                                    "handle_contact22", "handle_contact23", "valve_contact0"]

        robot_contact_list = ["link0_contact", "link1_contact", "link2_contact", "link3_contact", \
                                   "link4_contact", "link5_contact0", "link5_contact1", "link5_contact2", \
                                   "link6_contact", "link7_contact", "hand_contact", "finger0_contact", \
                                   "finger1_contact", "finger_contact0", "finger_contact1", "finger_contact2",\
                                   "finger_contact3", "finger_contact4", "finger_contact5", "finger_contact6",\
                                   "finger_contact7", "finger_contact8", "finger_contact9"]

        object_contact_list = ["handle_contact0", "handle_contact1", "handle_contact2", "handle_contact3", \
                                    "handle_contact4", "handle_contact5", "handle_contact6", "handle_contact7", \
                                    "handle_contact8", "handle_contact9", "valve_contact0",\
                                    "valve_contact0", "valve_contact1", "valve_contact2", "valve_contact3",\
                                    "valve_contact4", "valve_contact5", "valve_contact6", "valve_contact7",\
                                    "valve_contact8", "valve_contact9", "valve_contact10", "valve_contact11",\
                                    "valve_contact12", "valve_contact13", "valve_contact0"]

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

        self.controller.randomize_env(r, obj, self.init_angle, self.goal_angle, RL)
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

        end_effector = self.controller.get_ee_simple()

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


        ee = self.controller.get_ee_simple()
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
    def _info(self):
        info = {
            "collision": self.contact_done,
            "bound" : self.bound_done,
        }
        return info

    def _construct_action_space(self):
        action_space = 10
        action_low = np.array([-1.57079, -1.57079, -1.57079, -1.57079, -1.57079, -1.57079, -0.5, -0.5, -0.5, 0])
        action_high = np.array([1.57079, 1.57079, 1.57079, 1.57079, 1.57079, 1.57079, 0.5, 0.5, 0.5, 0.04])
        return gym.spaces.Box(low=action_low, high=action_high, dtype=np.float32)
    def _construct_observation_space(self):

        s = {
            'object': spaces.Box(shape=(1, 17), low=-np.inf, high=np.inf, dtype=np.float32),
            'q': spaces.Box(shape=(self.stack, self.dof), low=-1, high=1, dtype=np.float32),
            # 'dq': spaces.Box(shape=(self.stack, self.dof), low=-1, high=1, dtype=np.float32),
            'r6d': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            # 'r6d_dot': spaces.Box(shape=(self.stack, 6), low=-1, high=1, dtype=np.float_),
            'x_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
            # 'xdot_pos': spaces.Box(shape=(self.stack, 3), low=-np.inf, high=np.inf, dtype=np.float_),
        }

        return spaces.Dict(s)
    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        else:
            self.viewer.sync()
            sleep(0.002)

    def env_randomization(self):

        obj_list = ["handle", "valve"]
        radius_list = [0.119, 0.1]
        o_margin_list = [[[0], [0.149], [0]], [[0],[0],[-0.017]]]
        o = randint(0, 1)
        obj = obj_list[o]
        o_margin = o_margin_list[o]
        radius = radius_list[o]
        handle_quat_candidate = [[0.25192415, -0.64412663, 0.57897236, 0.4317709],
                                 [-0.49077636, 0.42062713, -0.75930974, 0.07523369],
                                 [0.474576307745582, -0.089013785474907, 0.275616460318178, 0.831197594392378],
                                 [0., -0.707, 0.707, 0.],
                                 [-0.46086475, -0.63305975, 0.39180338, 0.48304156],
                                 [-0.07865809, -0.89033475, 0.16254433, -0.41796684],
                                 [0.70738827, 0., 0., 0.70682518]]
        handle_pos_candidate = [[0.52, 0, 0.8],
                                [0.38, 0, 0.9],
                                [0.326, 0.232, 0.559],
                                [0.55, 0., 0.75],
                                [0.4, 0.3, 0.5],
                                [0.35, 0.45, 0.9],
                                [0.48, 0, 0.9]]
        valve_quat_candidate = [[0., 1., 0., 0.],
                                [-0.707, 0.707, 0., 0.],
                                [0., -0.707, 0.707, 0.],
                                [0., -0.707, 0.707, 0.],
                                [0., 0.707, - 0., 0.707],
                                [-0.707, 0.707, 0., 0.],
                                [0., 1., 0., 0.]]
        valve_pos_candidate = [[0.38, 0., 0.45],
                               [-0.2, 0.5, 0.6],
                               [0.28, 0., 0.7],
                               [0.38, 0., 0.5],
                               [0.48, 0., 0.55],
                               [0.3, 0.3, 0.6],
                               [0.3, 0.3, 0.6]]

        if obj == "handle":
            nobj = "valve"
            quat_candidate = handle_quat_candidate
            pos_candidate = handle_pos_candidate
        elif obj == "valve":
            nobj = "handle"
            quat_candidate = valve_quat_candidate
            pos_candidate = valve_pos_candidate

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



    def read_file(self):
        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dr_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dr = list(map(float, f_list))
        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dp_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dp = list(map(float, f_list))

        with open('/home/kist-robot2/catkin_ws/src/franka_emika_panda/build/devel/lib/franka_emika_panda/dy_heuristic.txt', 'r') as f:
            f_line = f.readline()  # 파일 한 줄 읽어오기
            f_list = f_line.split()  # 그 줄을 list에 저장

            self.dy = list(map(float, f_list))
