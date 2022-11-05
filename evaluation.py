# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import wandb

from level_replay.algo.policy import DDQN
from train_rainbow import eval_policy, generate_seeds

import argparse
from distutils.util import strtobool


def construct_class_from_dict(d):
    class Args:
        def __init__(self, dictionary):
            for k, v in dictionary.items():
                setattr(self, k, v)

    args = Args(d)

    return args


def evaluate(args):
    dictionary = torch.load(args.model_path)
    model_state_dict = dictionary["model_state_dict"]
    arg_dict = dictionary["args"]

    model_args = construct_class_from_dict(arg_dict)

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="off-policy-procgen",
        entity="ucl-dark",
        config=vars(model_args),
        tags=["ddqn", "procgen"] + (args.wandb_tags.split(",") if args.wandb_tags else []),
        group="Evaluations",
    )

    agent = DDQN(model_args)

    agent.Q.load_state_dict(model_state_dict)

    if args.test:
        eval_episode_rewards = eval_policy(model_args, agent, model_args.num_test_seeds)
        wandb.log(
            {
                "Test Evaluation Returns": np.mean(eval_episode_rewards),
            }
        )

    if args.each_train_level:
        train_eval_episode_rewards = []
        for seed in range(model_args.num_train_seeds):
            rewards = eval_policy(
                model_args, agent, model_args.num_test_seeds, start_level=seed, num_levels=1, seeds=[seed]
            )
            train_eval_episode_rewards.append([seed, np.mean(rewards)])
        table = wandb.Table(data=train_eval_episode_rewards, columns=["Train Level", "Evaluation Rewards"])
        wandb.log(
            {
                "Train Evaluations for Each Training Level": wandb.plot.bar(
                    table,
                    "Train Level",
                    "Evaluation Rewards",
                    title="Train Evaluations for Each Training Level",
                )
            }
        )

    else:
        seeds = generate_seeds(args.num_train_seeds)
        train_eval_episode_rewards = eval_policy(
            model_args,
            agent,
            model_args.num_test_seeds,
            start_level=0,
            num_levels=model_args.num_train_seeds,
            seeds=seeds,
        )
        wandb.log(
            {
                "Train Evaluation Returns": np.mean(train_eval_episode_rewards),
            }
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN")

    parser.add_argument("--model_path", default="model/model.tar", help="Path to pre-trained model")
    parser.add_argument(
        "--test", type=lambda x: bool(strtobool(x)), default=True, help="Whether to evaluate on unseen levels"
    )
    parser.add_argument(
        "--each_train_level",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="Whether to get score for each train level",
    )

    parser.add_argument("--no_cuda", type=lambda x: bool(strtobool(x)), default=False, help="disables gpu")

    args = parser.parse_args()

    evaluate(args)
