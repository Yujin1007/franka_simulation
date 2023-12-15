import numpy as np
import os
import copy

from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic
from fr3_envs.fr3_tqc import Fr3_tqc

import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc_ = nn.Linear(256, 1024)
        self.fc4 = nn.Linear(1024, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc_(x))
        x = F.relu(self.fc4(x))
        return x
    
HOME = os.getcwd()

def main():
    env = Fr3_tqc()
    env.env_rand = False
    env.rendering = True
    TRAIN = False

    max_timesteps = 1e6
    batch_size = 256
    policy_kwargs = dict(n_critics=5, n_quantiles=25)
    save_freq = 1e5
    models_dir = os.path.join(HOME, "log", "rpy", "handle_only5")
    models_subdir = os.path.join(models_dir, "7.0")
    episode_data = []
    save_flag = False

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    replay_buffer = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, policy_kwargs["n_quantiles"], policy_kwargs["n_critics"]).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=2,
                      discount=0.99,
                      tau=0.005,
                      target_entropy=-np.prod(env.action_space.shape).item())

    state = env.reset()
    episode_return = 0
    episode_timesteps = 0
    episode_num = 0

    if TRAIN:
        actor.train()
        for t in range(int(max_timesteps)):
            action = actor.select_action(state)


            next_state, reward, done, _ = env.step(action)
            episode_timesteps += 1

            replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_return += reward

            # Train agent after collecting sufficient data
            if t >= batch_size:
                trainer.train(replay_buffer, batch_size)
            if (t + 1) % save_freq == 0:
                save_flag = True
            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(
                    f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f}")
                # Reset environment
                state = env.reset()
                episode_data.append([episode_timesteps, episode_return, env.handle_angle])
                episode_return = 0
                episode_timesteps = 0
                episode_num += 1
            if save_flag:
                path = os.path.join(models_dir, str((t + 1) // save_freq))
                os.makedirs(path, exist_ok=True)
                # if not os.path.exists(path):
                #     os.makedirs(path)
                trainer.save(path)
                np.save(path + "reward", episode_data)
                save_flag = False

    else:
        trainer.load(models_subdir)
        # reset_agent.load(models_subdir)
        actor.eval()
        critic.eval()
        actor.training = False
        # reset_agent.training = False
        num_ep = 5

        for _ in range(num_ep):
            state = env.reset()
            done = False
            while not done:
                action = actor.select_action(state)

                next_state, reward, done, _ = env.step(action)
                state = next_state
                episode_return += reward
            print("episode :", env.episode_number, "goal angle :", env.required_angle, "handle angle :", env.handle_angle)

            print("time:",env.time_done, "  contact:",env.contact_done, "  bound:",env.bound_done,
                  "  goal:", env.goal_done, "  reset:",env.reset_done)

if __name__ == "__main__":
    main()
