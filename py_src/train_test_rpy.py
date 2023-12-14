import gym
import torch as th

from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tqc_savemodel import TQCsm

# from sb3_contrib.tqcsm.tqc_savemodel import TQCsm

import fr3_envs._fr3Env as _fr3Env
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
    MODEL_PATH = "./log/6d/TQC_2/step_900012"        # env ->_fr3Env.fr3_6d_test()
    # MODEL_PATH = "./log/3d/TQC_16/step_1000000.zip" # env  -> _fr3Env.fr3_3d_test()

    istrain = True
    isrendering = True
    israndomenv = True
    isheuristic = False
    # env = _fr3Env.fr3_6d_test()
    env = _fr3Env.fr3_6d_train()
    # env = _fr3Env.fr3_3d_test()
    env.env_rand = israndomenv
    env.rendering = isrendering
    policy_kwargs = dict(n_critics=2, n_quantiles=25)
    model = TQCsm("MultiInputPolicy",
                  env,
                  top_quantiles_to_drop_per_net=2,
                  verbose=1,
                  policy_kwargs=policy_kwargs,
                  save_path='/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/saved_model/step_{}',
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
        iteration = 600
        if isheuristic:  # run heuristic code with controller

            env.run(iteration)

        else:  # test trained model
            del model
            model = TQCsm.load(MODEL_PATH)
            save_theta = np.zeros([300,2])
            cnt = 0
            for i in range(iteration):
                done = False
                obs = env.reset("clk")

                while not done:

                    action, _states = model.predict(obs)
                    obs, rewards, done, info = env.step(action)

                    # env.render()
                print(np.round(env.max_rotation,3), np.round(env.data.qpos[-1],3))
                # np.save("./demonstration/manual_frame_rpy", env.rpyfromvalve_data)
                # np.save("./demonstration/manual_frame_xyz", env.xyzfromvalve_data)
                # np.save("./demonstration/manual_frame_grp", env.gripper_data)
                # np.save("./demonstration/rl_frame_rpy", env.rpyfromvalve_data)
                # np.save("./demonstration/rl_frame_xyz", env.xyzfromvalve_data)
                # np.save("./demonstration/rl_frame_grp", env.gripper_data)
                save_theta[i//2][env.episode_number % 2] = np.round(env.max_rotation,3)
                if i//2 == 0 and save_theta[i//2][env.episode_number % 2] != 0.0:
                    cnt += 1
                if cnt == 10:
                    break
            # print(save_theta)
            _save_theta = []
            for d in save_theta:
                if d[0] != 0.0:
                    _save_theta.append(d)
            print(_save_theta)
            np.save("./demonstration/clk_handle2", _save_theta)
if __name__ == "__main__":
    main()