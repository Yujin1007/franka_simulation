import os
import numpy as np
from sb3_contrib import TQC
from fr3_envs.fr3_full_action import Fr3_full_action

HOME = os.getcwd()

def main():
    total_timestep = 5e6
    MODEL_PATH = os.path.join("log", "parameters.zip")

    istrain =  False
    isrendering =True
    israndomenv = True
    isheuristic = False

    # Import the environment
    env = Fr3_full_action()
    env.env_rand = israndomenv
    env.rendering = isrendering
    # env.reset()
    # policy_kwargs = dict(features_extractor_class=CombinedExtractor, net_arch=[256, 128, 64],  n_critics=5, n_quantiles=25)
    policy_kwargs = dict(n_critics=5, n_quantiles=25)

    model = TQC("MultiInputPolicy",
                env,
                top_quantiles_to_drop_per_net=2,
                verbose=1,
                policy_kwargs=policy_kwargs,
                # save_path='/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/log/full_action/TQC_6/step_{}',
                # save_interval=1e5,
                tensorboard_log=os.path.join("log", "full_action"),
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
            os.chdir(HOME)
            model = TQC.load(MODEL_PATH)
            for _ in range(iteration):
                obs = env.reset()
                done = False

                while not done:
                    action, _ = model.predict(obs)
                    # print(action)
                    obs, _, done, _ = env.step(action)

                    env.render()
                print(np.round(env.handle_angle))
                # print(env.reward_accum)

if __name__ == "__main__":
    main()