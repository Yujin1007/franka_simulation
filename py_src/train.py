import os
import argparse

from models.tqc.tqc import TQC
from fr3_envs.fr3_tqc import Fr3_tqc
from models.classifier.classifier_tqc import Classifier
    
HOME = os.getcwd()

def main(args):
    env = Fr3_tqc()
    policy_kwargs = dict(n_critics=5, n_quantiles=25)    
    model = TQC(env, policy_kwargs)

    # Train mode
    if args.exec == "train":
        model.train()

    # Evaluation mode
    elif args.exec == "eval":
        model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec", help="Execution mode", default="eval", choices=["train", "eval"])
    args = parser.parse_args()

    main(args)
