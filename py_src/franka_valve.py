import os
import argparse

from models.tqc.tqc import TQC
from fr3_envs.fr3_tqc import Fr3_tqc
from models.classifier.classifier_tqc import Classifier
    
HOME = os.getcwd()

MAX_TIMESTEPS = 1e6
BATCH_SIZE = 256
SAVE_FREQ = 1e5

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
        max_timesteps = MAX_TIMESTEPS, 
        batch_size    = BATCH_SIZE, 
        save_freq     = SAVE_FREQ
    ):
    '''
    Train franka_valve

    - hyperparameters for an environment
    :param acc: Reward for difference of rpy (Negative), default: 3
    :param c: Reward for collision (Negative), default: 10
    :param b: Reward for joint boundary limit (Negative), default: 1
    :param gr: Reward for grasping an object (Positive), default: 1
    :param history: The length of a history to observe, default: 5
    :param object: Object to rotate, default: handle, choices: [handle, valve]

    - hyperparameters for a model
    :max_timesteps: Max timesteps, default: 1e6 (1,000,000)
    :batch_size: Batch size, default: 256
    :save_freq: Save frequency, default: 1e5 (100,000)
    '''
    check_valid_object(object)
    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=5, n_quantiles=25)
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
        max_timesteps = MAX_TIMESTEPS, 
        batch_size    = BATCH_SIZE, 
        save_freq     = SAVE_FREQ
    ):
    '''
    Evaluate franka_valve

    - hyperparameters for an environment
    :param acc: Reward for difference of rpy (Negative), default: 3
    :param c: Reward for collision (Negative), default: 10
    :param b: Reward for joint boundary limit (Negative), default: 1
    :param gr: Reward for grasping an object (Positive), default: 1
    :param history: The length of a history to observe, default: 5
    :param object: Object to rotate, default: handle, choices: [handle, valve]

    - hyperparameters for a model
    :max_timesteps: Max timesteps, default: 1e6 (1,000,000)
    :batch_size: Batch size, default: 256
    :save_freq: Save frequency, default: 1e5 (100,000)
    '''
    check_valid_object(object)
    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=5, n_quantiles=25)
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
