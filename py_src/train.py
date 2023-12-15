import os
import copy
import argparse

from tqc import structures, DEVICE
from tqc.trainer import Trainer
from tqc.structures import Actor, Critic
from tqc.train_and_test import train, eval
from fr3_envs.fr3_tqc import Fr3_tqc

import numpy as np
from models.classifier_tqc import Classifier
    
HOME = os.getcwd()

def main(args):
    env = Fr3_tqc()
    env.env_rand = False
    env.rendering = True

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

    # Train mode
    if args.exec == "train":
        train(actor, env, trainer, replay_buffer)
    # Evaluation mode
    elif args.exec == "eval":
        eval(actor, env, critic, trainer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exec", help="Execution mode", default="eval", choices=["train", "eval"])
    args = parser.parse_args()

    main(args)
