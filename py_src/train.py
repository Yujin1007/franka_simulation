# import gym
# import torch as th
#
# from torch import nn
# from gymnasium import spaces
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# import matplotlib.pyplot as plt
# from sb3_contrib.tqcsm.tqc_savemodel import TQCsm
#
# from stable_baselines3.common.env_util import make_vec_env
# import fr3Env
# from time import sleep
# import torch.nn as nn
# import torch.nn.functional as F
# from stable_baselines3 import PPO
# from stable_baselines3 import TD3
# from stable_baselines3.td3.policies import MlpPolicy
# from stable_baselines3.common.vec_env import DummyVecEnv
# import numpy as np
# # Parallel environments
#
#
#
#
# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: spaces.Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super().__init__(observation_space, features_dim=1)
#
#         extractors = {}
#
#         total_concat_size = 0
#         # We need to know size of the output of this extractor,
#         # so go over all the spaces and compute output feature sizes
#         for key, subspace in observation_space.spaces.items():
#             if key == "object":
#                 extractors[key] = nn.Sequential(nn.Flatten())
#                 total_concat_size += subspace.shape[1]
#             elif key == "q":
#                 extractors[key] = nn.Sequential(nn.Flatten(),nn.Linear(subspace.shape[0]*subspace.shape[1], 64))
#                 total_concat_size += 64
#             elif key == "r6d":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 64))
#                 total_concat_size += 64
#             elif key == "x_plan":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 32))
#                 total_concat_size += 32
#             elif key == "x_pos":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 32))
#                 total_concat_size += 32
#             elif key == "manipulability":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Sequential(nn.Flatten(), nn.Linear(subspace.shape[0] * subspace.shape[1], 16))
#                 total_concat_size += 16
#
#         self.extractors = nn.ModuleDict(extractors)
#
#         # Update the features dim manually
#         self._features_dim = total_concat_size
#
#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []
#
#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return th.cat(encoded_tensor_list, dim=1)
# class Classifier(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(input_size, 256)
#         self.fc2 = nn.Linear(256, 120)
#         self.fc3 = nn.Linear(120, 84)
#         self.fc4 = nn.Linear(84, output_size)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x
# def main():
#     # Parallel environments
#
#     total_timestep = 1e6
#
#     istrain = False
#     isrendering = True
#     israndomenv = False
#     isheuristic = False
#     env = fr3Env.fr3_test()
#     env.env_rand = israndomenv
#     env.rendering = isrendering
#     # env.reset()
#     policy_kwargs = dict(n_critics=2, n_quantiles=25)
#     # policy_kwargs = dict(
#     #     features_extractor_class=CustomCombinedExtractor, n_critics=2, n_quantiles=25
#     # )
#
#     # model = TQC("MultiInputPolicy", env, n_steps=n_step, policy_kwargs=policy_kwargs, verbose=1)
#     model = TQCsm("MultiInputPolicy",
#                   env,
#                   top_quantiles_to_drop_per_net=2,
#                   verbose=1,
#                   policy_kwargs=policy_kwargs,
#                   save_path='/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/saved_model/step_{}',
#                   save_interval=1e5,
#                   tensorboard_log="log/6d/",
#                   learning_starts=100,
#                   gamma=1,
#                   target_update_interval=100,
#                   # train_freq=(10, "episode")
#                   )
#
#
#     # model = PPO("MultiInputPolicy", env,tensorboard_log="log/",  learning_rate=1e-4,)
#
#     if istrain:
#         model.learn(total_timesteps=total_timestep)
#
#         model.save("tqc_franka")
#
#     # del model # remove to demonstrate saving and loading
#     else:
#         iteration = 8
#         if isheuristic:  # run heuristic code with controller
#
#             env.run(iteration)
#
#         else:  # test trained model
#             del model
#             # model = TQCsm.load("/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/log/TQC_2/model_success_999225.zip")
#
#             MODEL_PATH = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/log/TQC_14/step_490029.zip"
#             # MODEL_PATH = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/log/6d/TQC_2/step_900012"
#             model = TQCsm.load(MODEL_PATH)
#             for _ in range(iteration):
#                 obs = env.reset('clk')
#                 done = False
#                 torque = []
#                 while not done:
#                     action, _states = model.predict(obs)
#                     # action = np.array([0,0,0])
#                     obs, rewards, done, info = env.step(action)
#                     torque.append(env.data.ctrl[0:7].copy())
#                     env.render()
#                 print(np.round(env.max_rotation,3), np.round(env.data.qpos[-1],3))
#                 x = np.arange(len(torque))
#                 y = list(map(list, zip(*torque)))
#                 plt.plot(x,y[0],x,y[1],x,y[2],x,y[3],x,y[4],x,y[5],x,y[6])
#                 plt.show()
#
# if __name__ == "__main__":
#     main()
import gym
import torch as th

from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from sb3_contrib.tqcsm.tqc_savemodel import TQCsm

import _fr3Env
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# Parallel environments
class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
def main():
    # Parallel environments

    total_timestep = 1e6
    # MODEL_PATH = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/log/6d/TQC_2/step_900012"
    MODEL_PATH = "/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/log/3d/TQC_16/step_1000000.zip"

    istrain = False
    isrendering = False
    israndomenv = True
    isheuristic = False
    env = _fr3Env.fr3_3d_test()
    env.env_rand = israndomenv
    env.rendering = isrendering
    policy_kwargs = dict(n_critics=2, n_quantiles=25)
    model = TQCsm("MultiInputPolicy",
                  env,
                  top_quantiles_to_drop_per_net=2,
                  verbose=1,
                  policy_kwargs=policy_kwargs,
                  save_path='/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/saved_model2/step_{}',
                  save_interval=1e5,
                  tensorboard_log="log/6d/",
                  learning_starts=100,
                  gamma=1,
                  target_update_interval=100,
                  # train_freq=(10, "episode")
                  )



    if istrain:
        model.learn(total_timesteps=total_timestep)

        model.save("tqc_franka")

    # del model # remove to demonstrate saving and loading
    else:
        iteration = 200
        if isheuristic:  # run heuristic code with controller

            env.run(iteration)

        else:  # test trained model
            del model
            model = TQCsm.load(MODEL_PATH)
            save_theta = np.zeros([iteration//2,2])
            for i in range(iteration):
                done = False
                obs = env.reset("cclk")

                while not done:

                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)

                    # env.render()
                print(np.round(env.max_rotation,3), np.round(env.data.qpos[-2],3))
                # np.save("./demonstration/manual_frame_rpy", env.rpyfromvalve_data)
                # np.save("./demonstration/manual_frame_xyz", env.xyzfromvalve_data)
                # np.save("./demonstration/manual_frame_grp", env.gripper_data)
                # np.save("./demonstration/rl_frame_rpy", env.rpyfromvalve_data)
                # np.save("./demonstration/rl_frame_xyz", env.xyzfromvalve_data)
                # np.save("./demonstration/rl_frame_grp", env.gripper_data)
                save_theta[i//2][env.episode_number % 2] = max(abs(np.round(env.data.qpos[-2], 3)), abs(np.round(env.max_rotation,3)))
            # print(save_theta)
            np.save("./demonstration/cclk_valve2", save_theta)
if __name__ == "__main__":
    main()