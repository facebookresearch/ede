# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import time
from collections import deque
from typing import List

import numpy as np
import torch
import wandb

from level_replay.algo.ppo import MultiWorkerPPO
from level_replay import utils
from level_replay.arguments import parser
from level_replay.envs import make_lr_venv
from level_replay.storage import SimpleRolloutStorage
from level_replay.utils import ppo_normalise_reward

os.environ["OMP_NUM_THREADS"] = "1"

last_checkpoint_time = None


def train(args, seeds):
    global last_checkpoint_time
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    args.device = device
    args.arch = "simple"
    if "cuda" in device.type:
        print("Using CUDA\n")

    assert args.num_processes / args.num_workers == int(
        args.num_processes / args.num_workers
    ), "Must be able to divide num processes evenly"

    torch.set_num_threads(1)

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project="off-policy-procgen",
        entity="anon",
        config=vars(args),
        tags=["ppo"] + (args.wandb_tags.split(",") if args.wandb_tags else []),
        group=args.wandb_group,
    )

    utils.seed(args.seed)

    # Configure logging
    if args.xpid is None:
        args.xpid = "lr-%s" % time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.expandvars(os.path.expanduser(args.log_dir))

    checkpointpath = os.path.expandvars(os.path.expanduser("%s/%s/%s" % (log_dir, args.xpid, "model.tar")))

    # Configure actor envs
    start_level = 0
    if args.full_train_distribution:
        num_levels = 0
        level_sampler_args = None
        seeds = None
    else:
        num_levels = 1
        level_sampler_args = dict(
            num_actors=args.num_processes,
            strategy=args.level_replay_strategy,
            replay_schedule=args.level_replay_schedule,
            score_transform=args.level_replay_score_transform,
            temperature=args.level_replay_temperature,
            eps=args.level_replay_eps,
            rho=args.level_replay_rho,
            nu=args.level_replay_nu,
            alpha=args.level_replay_alpha,
            staleness_coef=args.staleness_coef,
            staleness_transform=args.staleness_transform,
            staleness_temperature=args.staleness_temperature,
        )
    envs, level_sampler = make_lr_venv(
        num_envs=args.num_processes,
        env_name=args.env_name,
        seeds=seeds,
        device=device,
        num_levels=num_levels,
        start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        use_sequential_levels=args.use_sequential_levels,
        level_sampler_args=level_sampler_args,
    )

    # is_minigrid = args.env_name.startswith("MiniGrid")

    rollouts = [
        SimpleRolloutStorage(
            args.num_steps,
            int(args.num_processes / args.num_workers),
            envs.observation_space.shape,
            envs.action_space,
        )
        for _ in range(args.num_workers)
    ]

    agent = MultiWorkerPPO(args, envs)

    def checkpoint():
        if args.disable_checkpoint:
            return
        logging.info("Saving checkpoint to %s", checkpointpath)
        torch.save(
            {
                "model_state_dict": agent.actor_critic.state_dict(),
                "optimizer_state_dict": agent.optimizer.state_dict(),
                "args": vars(args),
            },
            checkpointpath,
        )

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        obs, level_seeds = envs.reset()
    else:
        obs = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)

    for idx, rollouts_ in enumerate(rollouts):
        begin = int(idx * args.num_processes / args.num_workers)
        end = int((idx + 1) * args.num_processes / args.num_workers)
        rollouts_.obs[0].copy_(obs[begin:end, :, :, :])
        rollouts_.to(device)

    episode_rewards: deque = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes // args.num_workers

    count = 0
    for j in range(num_updates):
        agent.actor_critic.train()
        for step in range(args.num_steps):
            count += 1
            # Sample actions
            with torch.no_grad():
                obs_id = torch.cat([rollouts_.obs[step] for rollouts_ in rollouts])
                value, action, action_log_dist = agent.actor_critic.act(obs_id)
                action_log_prob = action_log_dist.gather(-1, action)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            reward = torch.from_numpy(reward)

            # Reset all done levels by sampling from level sampler
            for i, info in enumerate(infos):
                if "episode" in info.keys():
                    episode_reward = info["episode"]["r"]
                    episode_rewards.append(episode_reward)
                    wandb.log(
                        {
                            "Train Episode Returns": episode_reward,
                            "Train Episode Returns (normalised)": ppo_normalise_reward(
                                episode_reward, args.env_name
                            ),
                        },
                        step=count * args.num_processes,
                    )
                if level_sampler:
                    level_seeds[i][0] = info["level_seed"]

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )

            for idx, rollouts_ in enumerate(rollouts):
                begin = int(idx * args.num_processes / args.num_workers)
                end = int((idx + 1) * args.num_processes / args.num_workers)
                rollouts_.insert(
                    obs[begin:end],
                    action[begin:end],
                    action_log_prob[begin:end],
                    action_log_dist[begin:end],
                    value[begin:end],
                    reward[begin:end],
                    masks[begin:end],
                    bad_masks[begin:end],
                    level_seeds[begin:end],
                )

        for rollouts_ in rollouts:
            with torch.no_grad():
                obs_id = rollouts_.obs[-1]
                next_value = agent.actor_critic.get_value(obs_id).detach()

            rollouts_.compute_returns(next_value, args.gamma, args.gae_lambda)

        # Update level sampler
        if level_sampler:
            level_sampler.update_with_rollouts(rollouts[0])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        wandb.log({"Value Loss": value_loss}, step=count * args.num_processes)
        for rollouts_ in rollouts:
            rollouts_.after_update()
        if level_sampler:
            level_sampler.after_update()

        # Log stats every log_interval updates or if it is the last update
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            mean_eval_rewards = np.mean(evaluate(args, agent.actor_critic, args.num_test_seeds, device))

            mean_train_rewards = np.mean(
                evaluate(
                    args,
                    agent.actor_critic,
                    args.num_test_seeds,
                    device,
                    start_level=0,
                    num_levels=args.num_train_seeds,
                    seeds=seeds,
                )
            )

            wandb.log(
                {
                    "Test Evaluation Returns": mean_eval_rewards,
                    "Train Evaluation Returns": mean_train_rewards,
                    "Test Evaluation Returns (normalised)": ppo_normalise_reward(
                        mean_eval_rewards, args.env_name
                    ),
                    "Train Evaluation Returns (normalised)": ppo_normalise_reward(
                        mean_train_rewards, args.env_name
                    ),
                },
                step=count * args.num_processes,
            )

            if j == num_updates - 1:
                print(f"\nLast update: Evaluating on {args.final_num_test_seeds} test levels...\n  ")
                # logging.info(f"\nLast update: Evaluating on {args.num_test_seeds} test levels...\n  ")
                final_eval_episode_rewards = evaluate(
                    args, agent.actor_critic, args.final_num_test_seeds, device
                )

                mean_final_eval_episode_rewards = np.mean(final_eval_episode_rewards)
                median_final_eval_episide_rewards = np.median(final_eval_episode_rewards)

                print("Mean Final Evaluation Rewards: ", mean_final_eval_episode_rewards)
                print("Median Final Evaluation Rewards: ", median_final_eval_episide_rewards)

                wandb.log(
                    {
                        "Mean Final Evaluation Rewards": mean_final_eval_episode_rewards,
                        "Median Final Evaluation Rewards": median_final_eval_episide_rewards,
                    }
                )

        if args.save_model:
            print(f"Saving model to {args.model_path}")
            if "models" not in os.listdir():
                os.mkdir("models")
            torch.save(
                {
                    "model_state_dict": agent.actor_critic.state_dict(),
                    "args": vars(args),
                },
                args.model_path,
            )
            wandb.save(args.model_path)


def generate_seeds(num_seeds, base_seed=0):
    return [base_seed + i for i in range(num_seeds)]


def load_seeds(seed_path):
    seed_path = os.path.expandvars(os.path.expanduser(seed_path))
    seeds = open(seed_path).readlines()
    return [int(s) for s in seeds]


def evaluate(
    args,
    actor_critic,
    num_episodes,
    device,
    num_processes=1,
    deterministic=False,
    start_level=0,
    num_levels=0,
    seeds=None,
    level_sampler=None,
    progressbar=None,
):
    actor_critic.eval()

    if level_sampler:
        start_level = level_sampler.seed_range()[0]
        num_levels = 1

    eval_envs, level_sampler = make_lr_venv(
        num_envs=num_processes,
        env_name=args.env_name,
        seeds=seeds,
        device=device,
        num_levels=num_levels,
        start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler=level_sampler,
    )

    eval_episode_rewards: List[float] = []

    if level_sampler:
        obs, _ = eval_envs.reset()
    else:
        obs = eval_envs.reset()

    while len(eval_episode_rewards) < num_episodes:
        with torch.no_grad():
            _, action, _ = actor_critic.act(obs, deterministic=deterministic)

        obs, _, done, infos = eval_envs.step(action)

        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])
                if progressbar:
                    progressbar.update(1)

    eval_envs.close()
    if progressbar:
        progressbar.close()

    if args.verbose:
        print(
            "Last {} test episodes: mean/median reward {:.1f}/{:.1f}\n".format(
                len(eval_episode_rewards), np.mean(eval_episode_rewards), np.median(eval_episode_rewards)
            )
        )

    return eval_episode_rewards


if __name__ == "__main__":
    args = parser.parse_args()

    print(args)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
