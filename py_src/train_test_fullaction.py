import gym
import torch as th

from torch import nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor

from tqc_savemodel import TQCsm

from stable_baselines3.common.env_util import make_vec_env
import _fr3Env
from time import sleep
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3 import TD3
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.buffers import DictReplayBuffer_YJ
import numpy as np
# Parallel environments



def main():
    # Parallel environments

    total_timestep = 5e6
    MODEL_PATH="./log/full_action/TQC_5/step_900103.zip"
    istrain =  True
    isrendering =False
    israndomenv = True
    isheuristic = False
    env = _fr3Env.fr3_full_action()
    env.env_rand = israndomenv
    env.rendering = isrendering
    # env.reset()
    # policy_kwargs = dict(features_extractor_class=CombinedExtractor, net_arch=[256, 128, 64],  n_critics=5, n_quantiles=25)
    policy_kwargs = dict(n_critics=5, n_quantiles=25)

    model = TQCsm("MultiInputPolicy",
                  env,
                  top_quantiles_to_drop_per_net=2,
                  verbose=1,
                  policy_kwargs=policy_kwargs,
                  save_path='/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/log/full_action/TQC_6/step_{}',
                  save_interval=1e5,
                  tensorboard_log="log/full_action/",
                  learning_starts=100,
                  gamma=0.99,
                  target_update_interval=10,
                  # replay_buffer_class=DictReplayBuffer_YJ
                  # train_freq=(10, "episode")
                  )


    # model = PPO("MultiInputPolicy", env,tensorboard_log="log/",  learning_rate=1e-4,)

    if istrain:
        model.learn(total_timesteps=total_timestep)

        model.save("tqc_franka")

    # del model # remove to demonstrate saving and loading
    else:
        iteration = 8
        if isheuristic:  # run heuristic code with controller

            env.run(iteration)

        else:  # test trained model
            del model
            model = TQCsm.load(MODEL_PATH)
            for _ in range(iteration):
                obs = env.reset()
                done = False

                while not done:
                    action, _states = model.predict(obs)
                    # print(action)
                    obs, rewards, done, info = env.step(action)

                    env.render()
                print(np.round(env.handle_angle))
                # print(env.reward_accum)

if __name__ == "__main__":
    main()