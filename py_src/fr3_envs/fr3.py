import os
from abc import *
from time import sleep

import torch
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

import utils.tools as tools
import mujoco
from mujoco import viewer
import gym
from gym import spaces

# Constants
BODY = 1
JOINT = 3
GEOM = 5
MOTION_TIME_CONST = 10.
TASK_SPACE_TIME = 3+1+0.5

RL = 2
MANUAL = 1

HOME = os.getcwd()

from utils import find_libraries
rbdl_path, urdfreader_path = find_libraries.find_libraries()

try:
    from controller.full_action import controller
except ImportError as ie:
    print("Register rbdl and rbdl_urdfreader to PATH")
    print(f"rbdl path : {rbdl_path}")
    print(f"urdfreader path : {urdfreader_path}")
    exit()


def BringClassifier(path):
    classifier = torch.load(path)
    classifier.eval()
    return classifier

# TODO : Parent Class 만들어 공통되는 내용 정리, 차별점 있는 부분만 상속
# TODO : 경로 정리

class Fr3:
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self):
        self.k = 7  # for jacobian calculation
        self.dof = 9  # all joints (include gripper joint)

        self.model = self.import_model()
        self.data = mujoco.MjData(self.model)
        self.controller = controller.CController(self.k)
        self._torque = np.zeros(self.dof, dtype=np.float64)
        self.rendering = False
        self.stack = 5

    def import_model(self):
        model_path = os.path.join("..", "model", "franka_emika_panda", "scene_valve.xml")
        return mujoco.MjModel.from_xml_path(model_path)

    @abstractmethod
    def reset(self, direction=None):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def _reward(self, action):
        pass

    @abstractmethod
    def _done(self):
        pass

    def _info(self):
        info = {
            "collision": self.contact_done,
            "bound" : self.bound_done,
        }
        return info

    @abstractmethod
    def _construct_action_space(self):
        pass

    @abstractmethod
    def _construct_observation_space(self):
        pass

    def render(self):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(model=self.model, data=self.data)
        else:
            self.viewer.sync()
            sleep(0.002)

    @abstractmethod
    def env_randomization(self):
        pass

    def save_frame_data(self, ee):
        r = R.from_euler('xyz', ee[1][3:6], degrees=False)
        rpyfromvalve_rot = r.inv() * R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)
        ee_align = R.from_euler('z', 45, degrees=True)
        rpyfromvalve = (ee_align * rpyfromvalve_rot).as_matrix()

        xyzfromvalve_rot = (R.from_matrix(self.obj_rotation) * R.from_matrix(self.T_vv)).as_matrix()
        xyzfromvalve_rot = np.concatenate([xyzfromvalve_rot, [[0, 0, 0]]], axis=0)
        xyzfromvalve_rot = np.concatenate([xyzfromvalve_rot, [[self.obj_normal[0]], [self.obj_normal[1]], [self.obj_normal[2]], [1]]], axis=1)

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

class Fr3_with_model(Fr3):
    def __init__(self):
        super().__init__()
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

        os.chdir(HOME)
        self.classifier_clk = BringClassifier(os.path.join("classifier", "clk", "model.pt"))
        self.classifier_cclk = BringClassifier(os.path.join("classifier", "cclk", "model.pt"))

        # TODO : json 파일로 따로 저장
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

    @abstractmethod
    def _observation(self, end_effector, rpy_des):
        pass

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