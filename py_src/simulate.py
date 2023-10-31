#!/usr/bin/env python3
import numpy as np
from build.devel.lib import controller
import mujoco
from time import sleep

from mujoco import viewer
class fr3:
    def __init__(self) -> None:
        self.k = 7  # for jacobian calculation
        self.dof = 9  # all joints (include gripper joint)
        self.model_path = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/scene_valve.xml"
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.controller = controller.CController(self.k)
        self._torque = np.zeros(self.dof, dtype=np.float64)

        self.viewer = viewer.launch_passive(model=self.model, data=self.data)

        # self.scene_option = mujoco.MjvOption()
        # self.scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        self.duration = 380  # (seconds)
        self.framerate = 10  # (Hz)
    def run(self) -> None:
        # sleep(10)

        while self.viewer.is_running():#self.data.time < self.duration:
            self.viewer.sync()
            self.controller.read(self.data.time, self.data.qpos[0:self.dof], self.data.qvel[0:self.dof],
                                 self.model.opt.timestep, self.data.xpos.reshape(66,))
            # print(self.data.time)
            self.controller.control_mujoco()

            self._torque = self.controller.write()
            for i in range(self.dof):
                self.data.ctrl[i] = self._torque[i]

            mujoco.mj_step(self.model, self.data)
            sleep(0.002)
        self.viewer.close()

def main():
    panda = fr3()
    panda.run()


if __name__ == "__main__":
    main()
