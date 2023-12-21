import argparse
from franka_valve import franka_simulation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exec", help="Execution mode", type=str, default="eval", choices=["train", "eval"])
    parser.add_argument("--rw_acc", help="Negative reward for variation of output action", type=int, default=3)
    parser.add_argument("--rw_c", help="Negative reward for collision", type=int,default=1)
    parser.add_argument("--rw_b", help="Negative reward for joint boundary limit", type=int, default=1)
    parser.add_argument("--rw_gr", help="Positive reward for grasping an object", type=float, default=1.0)
    parser.add_argument("--history", help="The length of the history to observe", type=int, default=5)
    parser.add_argument("--object", help="Object to rotate", default="handle", choices=["handle", "valve"])

    parser.add_argument("--max_timesteps", help="Max timesteps", type=int, default=1e6)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=256)
    parser.add_argument("--save_freq", help="Save frequency", type=int, default=1e5)
    parser.add_argument("--n_critics", help="The number of critics", type=int, default=5)
    parser.add_argument("--n_quantiles", help="The number of quantiles", type=int, default=25)
    parser.add_argument("--model_index", help="Index of the model to load", type=str, default="default_model")
    parser.add_argument("--epochs_index", help="Index of the state to load", type=str, default="0.0")

    args = parser.parse_args()

    # Print Arguments
    print("------------ Arguments -------------")
    for key, value in vars(args).items():
        print(f"{key} : {value}")
    print("------------------------------------")

    if args.exec == "train":
        franka_simulation.train(rw_acc=args.rw_acc, rw_c=args.rw_c, rw_b=args.rw_b, rw_gr=args.rw_gr, history=args.history, object=args.object,
                                max_timesteps=args.max_timesteps, batch_size=args.batch_size, save_freq=args.save_freq, n_critics=args.n_critics, n_quantiles=args.n_quantiles)
    elif args.exec == "eval":
        franka_simulation.eval(rw_acc=args.rw_acc, rw_c=args.rw_c, rw_b=args.rw_b, rw_gr=args.rw_gr, history=args.history, object=args.object,
                               max_timesteps=args.max_timesteps, batch_size=args.batch_size, save_freq=args.save_freq, n_critics=args.n_critics, n_quantiles=args.n_quantiles,
                               model_index=args.model_index, epochs_index=args.epochs_index)