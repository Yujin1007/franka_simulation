import os
import json
from abc import *
from time import sleep

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

import mujoco
from mujoco import viewer

# Import requried libraries
from utils import find_libraries
rbdl_path, urdfreader_path = find_libraries.find_libraries()

try:
    from cpp_library import controller
except ImportError as ie:
    print("Register rbdl and rbdl_urdfreader to PATH")
    print("Command : ")
    print(f"export LD_LIBRARY_PATH={rbdl_path}:$LD_LIBRARY_PATH")
    print(f"export LD_LIBRARY_PATH={urdfreader_path}:$LD_LIBRARY_PATH")
    exit()

# Constants
BODY = 1
JOINT = 3
GEOM = 5
MOTION_TIME_CONST = 10.
TASK_SPACE_TIME = 3+1+0.5

RL = 2
MANUAL = 1

HOME = os.getcwd()

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

    # Read json file of contact information
    def read_contact_json(self, json_name):
        json_path = os.path.join(HOME, "fr3_envs", "jsons", "contacts", json_name)
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

        desired_contact_list = json_data["desired_contact_list"]
        desired_contact_list_finger = json_data["desired_contact_list_finger"]
        desired_contact_list_obj = json_data["desired_contact_list_obj"]
        robot_contact_list = json_data["robot_contact_list"]
        object_contact_list = json_data["object_contact_list"]

        return desired_contact_list, desired_contact_list_finger, desired_contact_list_obj, robot_contact_list, object_contact_list

    def import_model(self):
        model_path = os.path.join("franka_emika_panda", "scene_valve.xml")
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

    @abstractmethod
    def env_randomization(self):
        pass

    # Read json file of candidate information
    # Return quat_candidate, pos_candidate, nobj
    def read_candidate_json(self, obj, json_name):
        json_path = os.path.join(HOME, "fr3_envs", "jsons", "candidates", json_name)
        with open(json_path, 'r') as json_file:
            json_data = json.load(json_file)

        handle_quat_candidate = json_data["handle_quat_candidate"]
        handle_pos_candidate = json_data["handle_pos_candidate"]
        valve_quat_candidate = json_data["valve_quat_candidate"]
        valve_pos_candidate = json_data["valve_pos_candidate"]
        
        if obj == "handle":
            return handle_quat_candidate, handle_pos_candidate, "valve"
        elif obj == "valve":
            return valve_quat_candidate, valve_pos_candidate, "handle"
        
        return None

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