# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import deque
from test import evaluate

import numpy as np
import torch
import wandb

from level_replay import algo, utils
from level_replay.arguments import parser
from level_replay.envs import make_lr_venv
from level_replay.model import model_for_env_name
from level_replay.storage import RolloutStorage
from level_replay.utils import ppo_normalise_reward, min_max_normalise_reward

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["WANDB_API_KEY"] = "anon"


def train(args, seeds):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if "cuda" in device.type:
        print("Using CUDA\n")

    torch.set_num_threads(1)

    wandb.init(
        settings=wandb.Settings(start_method="fork"),
        project=args.wandb_project,
        entity="anon",
        config=vars(args),
        tags=["ppo"] + (args.wandb_tags.split(",") if args.wandb_tags else []),
        group=args.wandb_group,
    )
    wandb.run.name = f"ppo-{args.env_name}-{args.num_train_seeds}-levels"

    utils.seed(args.seed)

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

    actor_critic = model_for_env_name(args, envs)
    actor_critic.to(device)

    rollouts = RolloutStorage(
        args.num_steps,
        args.num_processes,
        envs.observation_space.shape,
        envs.action_space,
        actor_critic.recurrent_hidden_state_size,
    )

    agent = algo.PPO(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm,
        env_name=args.env_name,
    )

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        obs, level_seeds = envs.reset()
    else:
        obs = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    episode_rewards: deque = deque(maxlen=10)
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    count = 0
    for j in range(num_updates):
        actor_critic.train()
        for step in range(args.num_steps):
            count += 1
            # Sample actions
            with torch.no_grad():
                obs_id = rollouts.obs[step]
                value, action, action_log_dist, recurrent_hidden_states = actor_critic.act(
                    obs_id, rollouts.recurrent_hidden_states[step], rollouts.masks[step]
                )
                action_log_prob = action_log_dist.gather(-1, action)

            # Observe reward and next obs
            obs, reward, done, infos = envs.step(action)
            reward = torch.from_numpy(reward)

            # Reset all done levels by sampling from level sampler
            for i, info in enumerate(infos):
                if "episode" in info.keys():
                    episode_reward = info["episode"]["r"]
                    episode_rewards.append(episode_reward)
                    ppo_normalised_reward = ppo_normalise_reward(episode_reward, args.env_name)
                    min_max_normalised_reward = min_max_normalise_reward(episode_reward, args.env_name)
                    wandb.log(
                        {
                            "Train Episode Returns": episode_reward,
                            "Train Episode Returns (normalised)": ppo_normalised_reward,
                            "Train Episode Returns (ppo normalised)": ppo_normalised_reward,
                            "Train Episode Returns (min-max normalised)": min_max_normalised_reward,
                        },
                        step=count * args.num_processes,
                    )
                    if args.log_per_seed_stats:
                        plot_level_returns(level_seeds, episode_reward, i, step=count * args.num_processes)
                if level_sampler:
                    level_seed = info["level_seed"]
                    if level_seeds[i][0] != level_seed:
                        level_seeds[i][0] = level_seed
                        if args.log_per_seed_stats:
                            new_episode(value, level_seed, i, step=count * args.num_processes)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if "bad_transition" in info.keys() else [1.0] for info in infos]
            )

            rollouts.insert(
                obs,
                recurrent_hidden_states,
                action,
                action_log_prob,
                action_log_dist,
                value,
                reward,
                masks,
                bad_masks,
                level_seeds,
            )

        with torch.no_grad():
            obs_id = rollouts.obs[-1]
            next_value = actor_critic.get_value(
                obs_id, rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]
            ).detach()

        rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)

        # Update level sampler
        if level_sampler:
            level_sampler.update_with_rollouts(rollouts)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        wandb.log({"Value Loss": value_loss}, step=count * args.num_processes)
        rollouts.after_update()
        if level_sampler:
            level_sampler.after_update()

        # Log stats every log_interval updates or if it is the last update
        if (j % args.log_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
            mean_eval_rewards = np.mean(evaluate(args, actor_critic, args.num_test_seeds, device))

            mean_train_rewards = np.mean(
                evaluate(
                    args,
                    actor_critic,
                    args.num_test_seeds,
                    device,
                    start_level=0,
                    num_levels=args.num_train_seeds,
                    seeds=seeds,
                )
            )
            test_ppo_normalised_reward = ppo_normalise_reward(mean_eval_rewards, args.env_name)
            train_ppo_normalised_reward = ppo_normalise_reward(mean_train_rewards, args.env_name)
            test_min_max_normalised_reward = min_max_normalise_reward(mean_eval_rewards, args.env_name)
            train_min_max_normalised_reward = min_max_normalise_reward(mean_train_rewards, args.env_name)
            wandb.log(
                {
                    "Test Evaluation Returns": mean_eval_rewards,
                    "Train Evaluation Returns": mean_train_rewards,
                    "Generalization Gap:": mean_train_rewards - mean_eval_rewards,
                    "Test Evaluation Returns (normalised)": test_ppo_normalised_reward,
                    "Train Evaluation Returns (normalised)": train_ppo_normalised_reward,
                    "Test Evaluation Returns (ppo normalised)": test_ppo_normalised_reward,
                    "Train Evaluation Returns (ppo normalised)": train_ppo_normalised_reward,
                    "Test Evaluation Returns (min-max normalised)": test_min_max_normalised_reward,
                    "Train Evaluation Returns (min-max normalised)": train_min_max_normalised_reward,
                },
                step=count * args.num_processes,
            )

            if j == num_updates - 1:
                print(f"\nLast update: Evaluating on {args.final_num_test_seeds} test levels...\n  ")
                final_eval_episode_rewards = evaluate(args, actor_critic, args.final_num_test_seeds, device)

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
                    "model_state_dict": actor_critic.state_dict(),
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


def new_episode(value, level_seed, i, step):
    wandb.log({f"Start State Value Estimate for Level {level_seed}": value[i].item()}, step=step)


def plot_level_returns(level_seeds, episode_reward, i, step):
    seed = level_seeds[i][0].item()
    wandb.log({f"Empirical Return for Level {seed}": episode_reward}, step=step)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds)

    train(args, train_seeds)
