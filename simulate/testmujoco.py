import mujoco
from mujoco import viewer
xml = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <body name="box_and_sphere" euler="0 0 -30">
      <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
      <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
      <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""
# xml = """
# <mujoco>
#     <option gravity="0 0 10"/>
#       <worldbody>
#         <light name="top" pos="0 0 1"/>
#         <body name="box_and_sphere" euler="0 0 -30">
#           <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
#           <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
#           <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
#         </body>
#       </worldbody>
# </mujoco>
# """
# model = mujoco.MjModel.from_xml_path("/home/kist-robot2/catkin_ws/src/franka_emika_panda/model/franka_emika_panda/scene_valve.xml")
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model)

viewer = viewer.launch_passive(model= model, data= data)

# enable joint visualization option:
scene_option = mujoco.MjvOption()
scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

duration = 3.8  # (seconds)
framerate = 10  # (Hz)

# Simulate and display video.
while (1):#data.time < duration:
    mujoco.mj_step(model, data)
    if data.time * framerate <= duration * framerate:
        renderer.update_scene(data, scene_option=scene_option)
        viewer.sync()
    print(data.time)