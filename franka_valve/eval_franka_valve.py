from franka_valve import check_valid_object, PACKAGE_DIR
from franka_valve.models.tqc.tqc import TQC
from franka_valve.fr3_envs.fr3_tqc import Fr3_tqc

def get_eval_model_path(model_index):
    if model_index == "default_model":
        return "default_model"
    
    try:
        int_index = int(model_index)
        return f"model_{int_index}"
    except ValueError:
        raise ValueError("Cannot change an input index to integer.")

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
    :param model_index: Index of the model to load
    :param epochs_index: Index of the state (save_freq * epochs_index) to load
    '''
    check_valid_object(object)
    model_path = get_eval_model_path(model_index)

    env = Fr3_tqc(rw_acc, rw_c, rw_b, rw_gr, history, object)
    policy_kwargs = dict(n_critics=n_critics, n_quantiles=n_quantiles)
    model = TQC(env, policy_kwargs, max_timesteps, batch_size, save_freq, model_path, epochs_index)

    model.eval()