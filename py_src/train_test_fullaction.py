import os
import argparse
import numpy as np

from sb3_contrib import TQC
from fr3_envs.fr3_full_action import Fr3_full_action

HOME = os.getcwd()
TOTAL_TIMESTEP = 5e6
ITERATION = 8

def eval(env, MODEL_PATH):
    model = TQC.load(MODEL_PATH)
    for _ in range(ITERATION):
        obs = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs)
            obs, _, done, _ = env.step(action)

            env.render()
        print(np.round(env.handle_angle))

def main(args):
    MODEL_PATH = os.path.join("log", "parameters.zip")
    isheuristic = False

    # Import the environment
    env = Fr3_full_action()
    policy_kwargs = dict(n_critics=5, n_quantiles=25)

    if args.exec == "train":
        env.env_rand = True
        env.rendering = False

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

        model.learn(total_timesteps=TOTAL_TIMESTEP)
        model.save("tqc_franka")

    # del model # remove to demonstrate saving and loading
    elif args.exec == "eval":
        env.env_rand = True
        env.rendering = False

        if isheuristic:  # run heuristic code with controller
            env.run(ITERATION)

        else:  # test trained model
            eval(env, MODEL_PATH)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec", type=str, default="eval", choices=["train", "eval"])
    args = parser.parse_args()

    main(args)