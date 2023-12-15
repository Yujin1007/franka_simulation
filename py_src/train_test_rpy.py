import os
import argparse
from utils.tqc_savemodel import TQCsm

from fr3_envs.fr3_6d_train import Fr3_6d_train
from fr3_envs.fr3_6d_test import Fr3_6d_test
from fr3_envs.fr3_3d_test import Fr3_3d_test

import numpy as np
from models.classifier.classifier_rpy import Classifier

HOME = os.getcwd()
    
def main(args):
    # Parallel environments
    total_timestep = 1e6

    # Import model
    if args.env[:2] == "6d":
        MODEL_PATH = os.path.join(HOME, "log", "6d", "TQC_2", "step_900012.zip")
    elif args.env[:2] == "3d":
        MODEL_PATH = os.path.join(HOME, "log", "3d", "TQC_16", "step_1000000.zip")

    istrain = True
    isrendering = True
    israndomenv = True
    isheuristic = False

    # Import the environment
    if args.env == "6d_train":
        env = Fr3_6d_train()
    elif args.env == "6d_test":
        env = Fr3_6d_test()
    elif args.env == "3d_test":
        env = Fr3_3d_test()

    env.env_rand = israndomenv
    env.rendering = isrendering
    policy_kwargs = dict(n_critics=2, n_quantiles=25)
    model = TQCsm("MultiInputPolicy",
                  env,
                  top_quantiles_to_drop_per_net=2,
                  verbose=1,
                  policy_kwargs=policy_kwargs,
                #   save_path='/home/kist-robot2/catkin_ws/src/franka_emika_panda/py_src/saved_model/step_{}',
                #   save_interval=1e5,
                  tensorboard_log=os.path.join("log", "6d") if args.env[:2] == "6d" else os.path.join("log", "3d"),
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

                    action, _ = model.predict(obs)
                    obs, _, done, _ = env.step(action)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="Environment", type=str, default="6d_test", choices=["6d_train", "6d_test", "3d_test"])
    args = parser.parse_args()

    main(args)