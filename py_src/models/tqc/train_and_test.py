import os
import numpy as np

# Constants
MAX_TIMESTEPS = 1e6
BATCH_SIZE = 256
SAVE_FREQ = 1e5

NUM_EP = 6

HOME = os.getcwd()
MODELS_DIR = os.path.join(HOME, "log", "rpy", "handle_only5")
MODELS_SUBDIR = os.path.join(MODELS_DIR, "7.0")

class TQC:
    def __init__(self, actor, env, trainer, critic, replay_buffer):
        super(TQC, self).__init__()
        self.actor = actor
        self.env = env
        self.trainer = trainer
        self.critic = critic
        self.replay_buffer = replay_buffer

    def train(self):
        episode_data = []
        save_flag = False

        state = self.env.reset()
        episode_return = 0
        episode_timesteps = 0
        episode_num = 0

        self.actor.train()
        for t in range(int(MAX_TIMESTEPS)):
            action = self.actor.select_action(state)

            next_state, reward, done, _ = self.env.step(action)
            episode_timesteps += 1

            self.replay_buffer.add(state, action, next_state, reward, done)

            state = next_state
            episode_return += reward

            # Train agent after collecting sufficient data
            if t >= BATCH_SIZE:
                self.trainer.train(self.replay_buffer, BATCH_SIZE)
            if (t + 1) % SAVE_FREQ == 0:
                save_flag = True

            if done:
                # +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
                print(f"Total T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_timesteps} Reward: {episode_return:.3f}")
                # Reset environment
                state = self.env.reset()
                episode_data.append([episode_timesteps, episode_return, self.env.handle_angle])
                episode_return = 0
                episode_timesteps = 0
                episode_num += 1
                
            if save_flag:
                path = os.path.join(MODELS_DIR, str((t + 1) // SAVE_FREQ))
                os.makedirs(path, exist_ok=True)
                if not os.path.exists(path):
                    os.makedirs(path)
                self.trainer.save(path)
                np.save(os.path.join(path, "reward"), episode_data)
                save_flag = False

    def eval(self):
        self.trainer.load(MODELS_SUBDIR)
        # reset_agent.load(models_subdir)
        self.actor.eval()
        self.critic.eval()
        self.actor.training = False
        # reset_agent.training = False
        episode_return = 0

        for _ in range(NUM_EP):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.actor.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_return += reward

            print(f"episode: {self.env.episode_number}, goal_angle: {self.env.required_angle}, handle_angle: {self.env.handle_angle}")
            print(f"time: {self.env.time_done}, contact: {self.env.contact_done}, bound: {self.env.bound_done}, goal: {self.env.goal_done}, reset: {self.env.reset_done}")