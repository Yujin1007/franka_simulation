import os
import argparse

from models.tqc.tqc import TQC
from fr3_envs.fr3_tqc import Fr3_tqc
    
HOME = os.getcwd()

def check_valid_object(object):
    valid_values = ["handle", "valve"]

    if object not in valid_values:
        raise ValueError(f"Invalid object. Valid values are {valid_values}")

# Train mode
def train_franka_valve(
        rw_acc        = 3,
        rw_c          = 10,
        rw_b          = 1,
        rw_gr         = 1,
        history       = 5,
        object        = "handle",
        max_timesteps = 1e6, 
        batch_size    = 256, 
        save_freq     = 1e5,
        n_critics     = 5,
        n_quantiles   = 25
    ):
    '''
    Train franka_valve

    @ hyperparameters for an environment
    :param acc: Reward for difference of rpy (Reward)
    :param c: Reward for collision (Penalty)
    :param b: Reward for joint boundary limit (Penalty)
    :param gr: Reward for grasping an object (Reward)
    :param history: The length of a history to observe
    :param object: Object to rotate, choices: [handle, valve]

    @ hyperparameters for a model
    :param max_timesteps: Max timesteps
    :param batch_size: Batch size
    :param save_freq: Save frequency
    :param n_critics: The number of critics
    :param n_quantiles: The number of quantiles
    '''
    check_valid_object(object)
    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=n_critics, n_quantiles=n_quantiles)
    model = TQC(env, policy_kwargs, max_timesteps, batch_size, save_freq)
    model.train()

# Evaluation mode
def eval_franka_valve(
        rw_acc        = 3,
        rw_c          = 10,
        rw_b          = 1,
        rw_gr         = 1,
        history       = 5,
        object        = "handle",
        max_timesteps = 1e6, 
        batch_size    = 256, 
        save_freq     = 1e5,
        n_critics     = 5,
        n_quantiles   = 25
    ):
    '''
    Evaluate franka_valve

    @ hyperparameters for an environment
    :param acc: Reward for difference of rpy (Reward)
    :param c: Reward for collision (Penalty)
    :param b: Reward for joint boundary limit (Penalty)
    :param gr: Reward for grasping an object (Reward)
    :param history: The length of a history to observe
    :param object: Object to rotate, choices: [handle, valve]

    @ hyperparameters for a model
    :param max_timesteps: Max timesteps
    :param batch_size: Batch size
    :param save_freq: Save frequency
    :param n_critics: The number of critics
    :param n_quantiles: The number of quantiles
    '''
    check_valid_object(object)
    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=n_critics, n_quantiles=n_quantiles)
    model = TQC(env, policy_kwargs, max_timesteps, batch_size, save_freq)
    model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exec", help="Execution mode", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--rw_acc", help="Reward for difference of rpy", type=int, default=3)
    parser.add_argument("--rw_c", help="Reward for collision", type=int,default=1)
    parser.add_argument("--rw_b", help="Reward for joint boundary limit", type=int, default=1)
    parser.add_argument("--rw_gr", help="Reward for grasping an object", type=float, default=1.0)
    parser.add_argument("--history", help="The length of a history to observe", type=int, default=5)
    parser.add_argument("--object", help="Object to rotate", default="handle", choices=["handle", "valve"])

    args = parser.parse_args()

    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")

    if args.exec == "train":
        train_franka_valve(rw_acc=args.rw_acc, rw_c=args.rw_c, rw_b=args.rw_b, rw_gr=args.rw_gr, history=args.history, object=args.object)
    elif args.exec == "eval":
        eval_franka_valve(rw_acc=args.rw_acc, rw_c=args.rw_c, rw_b=args.rw_b, rw_gr=args.rw_gr, history=args.history, object=args.object)
