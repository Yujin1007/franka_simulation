import os
import json

from franka_valve import PACKAGE_DIR
from franka_valve.models.tqc.tqc import TQC
from franka_valve.fr3_envs.fr3_tqc import Fr3_tqc

def check_valid_object(object):
    valid_values = ["handle", "valve"]

    if object not in valid_values:
        raise ValueError(f"Invalid object. Valid values are {valid_values}")

# Train mode
def get_train_model_path():
    base_path = os.path.join(PACKAGE_DIR, "models", "tqc", "model")
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
def train(
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
    
    json_path = os.path.join(PACKAGE_DIR, "models", "tqc", "model", model_path)
    with open(os.path.join(json_path, "hyperparameters.json"), 'w') as f:
        json.dump(hyperparameters, f, ensure_ascii=False, indent=4)

    # Load env & model
    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=n_critics, n_quantiles=n_quantiles)
    model = TQC(env, policy_kwargs, max_timesteps, batch_size, save_freq, model_path, epochs_index="-1.0")

    model.train()

# Evaluation mode
def eval(
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
    :param model_index: Index of the model to load
    :param epochs_index: Index of the state (save_freq * epochs_index) to load
    '''
    check_valid_object(object)
    model_path = get_eval_model_path(model_index)

    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=n_critics, n_quantiles=n_quantiles)
    model = TQC(env, policy_kwargs, max_timesteps, batch_size, save_freq, model_path, epochs_index)

    model.eval()