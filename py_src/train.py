import os
import copy
import argparse

from models.tqc import structures, DEVICE
from models.tqc.trainer import Trainer
from models.tqc.structures import Actor, Critic
from models.tqc.train_and_test import TQC
from fr3_envs.fr3_tqc import Fr3_tqc

import numpy as np
from models.classifier.classifier_tqc import Classifier
    
HOME = os.getcwd()

def main(args):
    env = Fr3_tqc()
    policy_kwargs = dict(n_critics=5, n_quantiles=25)

    state_dim = env.observation_space.shape
    action_dim = env.action_space.shape[0]

    replay_buffer = structures.ReplayBuffer(state_dim, action_dim)
    actor = Actor(state_dim, action_dim).to(DEVICE)
    critic = Critic(state_dim, action_dim, policy_kwargs["n_quantiles"], policy_kwargs["n_critics"]).to(DEVICE)
    critic_target = copy.deepcopy(critic)

    trainer = Trainer(actor=actor,
                      critic=critic,
                      critic_target=critic_target,
                      top_quantiles_to_drop=2,
                      discount=0.99,
                      tau=0.005,
                      target_entropy=-np.prod(env.action_space.shape).item())
    
    model = TQC(actor, env, trainer, critic, replay_buffer)

    # Train mode
    if args.exec == "train":
        env.env_rand = True
        env.rendering = False
        model.train()

    # Evaluation mode
    elif args.exec == "eval":
        env.env_rand = False
        env.rendering = True
        model.eval()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec", help="Execution mode", default="eval", choices=["train", "eval"])
    args = parser.parse_args()

    main(args)
