import os
import json
import argparse

from models.tqc.tqc import TQC
from fr3_envs.fr3_tqc import Fr3_tqc
    
HOME = os.getcwd()

def check_valid_object(object):
    valid_values = ["handle", "valve"]

    if object not in valid_values:
        raise ValueError(f"Invalid object. Valid values are {valid_values}")

# Train mode
def get_train_model_path():
    base_path = os.path.join(HOME, "models", "tqc", "model")
    models = os.listdir(base_path)
    models.sort()

    try:
        model_name = f"model_{int(models[-1][-1])+1}"
        os.makedirs(os.path.join(base_path, model_name), exist_ok=True)
        return model_name
    except ValueError:
        model_name = "model_1"
        os.makedirs(os.path.join(base_path, model_name), exist_ok=True)
        return model_name

# Evaluation mode
def get_eval_model_path(model_index):
    if model_index == "default_model":
        return "default_model"
    
    try:
        int_index = int(model_index)
        return f"model_{int_index}"
    except ValueError:
        raise ValueError("Cannot change an input index to integer.")
    
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

    Hyperparameters for environments
    --------------------------------
    :param acc: Negative reward for variation of output action
    :param c: Negative reward for collision
    :param b: Negative reward for joint boundary limit
    :param gr: Positive reward for grasping an object
    :param history: The length of the history to observe
    :param object: Object to rotate, choices: [handle, valve]

    Hyperparameters for models
    --------------------------
    :param max_timesteps: Max timesteps
    :param batch_size: Batch size
    :param save_freq: Save frequency
    :param n_critics: The number of critics
    :param n_quantiles: The number of quantiles
    '''
    check_valid_object(object)
    model_path = get_train_model_path()

    # Save hyperparameters
    env_hyperparameters = {
                            "rw_acc" : rw_acc,
                            "rw_c" : rw_c,
                            "rw_b" : rw_b,
                            "rw_gr" : rw_gr,
                            "history" : history,
                            "object" : object
                          }
    model_hyperparameters = {
                                "max_timesteps" : max_timesteps,
                                "batch_size" : batch_size,
                                "save_freq" : save_freq,
                                "n_critics" : n_critics,
                                "n_quantiles" : n_quantiles
                            }
    
    hyperparameters = {
                        "environments" : env_hyperparameters,
                        "models" : model_hyperparameters
                      }
    
    os.chdir(os.path.join(HOME, "models", "tqc", "model", model_path))
    with open("hyperparameters.json", 'w') as f:
        json.dump(hyperparameters, f, ensure_ascii=False, indent=4)
    os.chdir(HOME)

    # Load env & model
    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=n_critics, n_quantiles=n_quantiles)
    model = TQC(env, policy_kwargs, max_timesteps, batch_size, save_freq, model_path, epochs_index="-1.0")

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
        n_quantiles   = 25,
        model_index   = "default_model",
        epochs_index  = "0.0"
    ):
    '''
    Evaluate franka_valve

    Hyperparameters for environments
    --------------------------------
    :param acc: Negative reward for variation of output action
    :param c: Negative reward for collision
    :param b: Negative reward for joint boundary limit
    :param gr: Positive reward for grasping an object
    :param history: The length of the history to observe
    :param object: Object to rotate, choices: [handle, valve]

    Hyperparameters for models
    --------------------------
    :param max_timesteps: Max timesteps
    :param batch_size: Batch size
    :param save_freq: Save frequency
    :param n_critics: The number of critics
    :param n_quantiles: The number of quantiles
    '''
    check_valid_object(object)
    model_path = get_eval_model_path(model_index)

    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=n_critics, n_quantiles=n_quantiles)
    model = TQC(env, policy_kwargs, max_timesteps, batch_size, save_freq, model_path, epochs_index)

    model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exec", help="Execution mode", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--rw_acc", help="Negative reward for variation of output action", type=int, default=3)
    parser.add_argument("--rw_c", help="Negative reward for collision", type=int,default=1)
    parser.add_argument("--rw_b", help="Negative reward for joint boundary limit", type=int, default=1)
    parser.add_argument("--rw_gr", help="Positive reward for grasping an object", type=float, default=1.0)
    parser.add_argument("--history", help="The length of the history to observe", type=int, default=5)
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
