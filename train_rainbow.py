# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from ast import arg
import logging
import os
from collections import deque
from typing import List
import time
import copy

import numpy as np
import torch

import wandb
from baselines import logger

from torch import nn
from level_replay.algo.dqn import init_

from level_replay import utils
from level_replay.algo.buffer import make_buffer, RolloutStorage
from level_replay.algo.policy import DQNAgent, ATCAgent
from level_replay.dqn_args import parser
from level_replay.envs import make_dqn_lr_venv
from level_replay.utils import ppo_normalise_reward, min_max_normalise_reward

from scipy.stats import skew, kurtosis
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["WANDB_API_KEY"] = "anon"
# os.environ["WANDB_BASE_URL"] = "https://api.fairwandb.ai"
# os.environ["WANDB_API_KEY"] = "092a14187f6f01d8d2df67e8145ed4b16ba8bc9d"


def train(args, seeds):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device("cuda:0" if args.cuda else "cpu")
    if "cuda" in args.device.type:
        print("Using CUDA\n")
    args.optimizer_parameters = {"lr": args.learning_rate, "eps": args.adam_eps}
    args.seeds = seeds

    args.sge_job_id = int(os.environ.get("JOB_ID", -1))
    args.sge_task_id = int(os.environ.get("SGE_TASK_ID", -1))
    args.PLR = False

    for k, v in vars(args).items():
        print(' ' * 26 + k + ': ' + str(v))

    torch.set_num_threads(1)

    utils.seed(args.seed)

    name =  (
        f"dqn-{args.env_name}-{args.num_train_seeds}levels"
        + f"{'-PER' if args.PER else ''}"
        + f"{'-dueling' if args.dueling else ''}"
        + f"{'-qrdqn' if args.qrdqn else ''}"
        + f"{'-c51' if args.c51 else ''}"
        + f"{'-drq' if args.drq else ''}"
        + f"{'-autodrq' if args.autodrq else ''}"
        + f"{'-atc' if args.atc else ''}"
        + f"{'-seed' if args.seed else ''}"
        + '-' + args.exp_name
    )

    if not args.wandb:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        # settings=wandb.Settings(start_method="fork"),
        project="dqn_procgen",
        entity="ydj",
        name=name,
        config=args,
        # tags=["ddqn", "procgen"] + (args.wandb_tags.split(",") if args.wandb_tags else []),
        group=None,
    )
    os.environ["WANDB_BASE_URL"] = "https://api.fairwandb.ai"
    os.environ["WANDB_API_KEY"] = "092a14187f6f01d8d2df67e8145ed4b16ba8bc9d"

    num_levels = 1
    level_sampler_args = dict(
        num_actors=args.num_processes,
        strategy=args.level_replay_strategy,
    )
    envs, level_sampler = make_dqn_lr_venv(
        num_envs=args.num_processes,
        env_name=args.env_name,
        seeds=seeds,
        device=args.device,
        num_levels=num_levels,
        start_level=args.start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        use_sequential_levels=args.use_sequential_levels,
        level_sampler_args=level_sampler_args,
        attach_task_id=args.attach_task_id
    )

    if args.atc:
        args.drq = True
        agent = ATCAgent(args, envs)
    else:
        agent = DQNAgent(args, envs)

    replay_buffer = make_buffer(args, envs)

    level_seeds = torch.zeros(args.num_processes)
    if level_sampler:
        state, level_seeds = envs.reset()
    else:
        state = envs.reset()
    level_seeds = level_seeds.unsqueeze(-1)

    if args.autodrq:
        rollouts = RolloutStorage(256, args.num_processes, envs.observation_space.shape, envs.action_space)
        rollouts.obs[0].copy_(state)
        rollouts.to(args.device)

    estimates = [0 for _ in range(args.num_train_seeds)]
    returns = [0 for _ in range(args.num_train_seeds)]
    gaps = [0 for _ in range(args.num_train_seeds)]

    episode_reward = 0

    state_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
    reward_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
    action_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
    expect_new_seed: List[bool] = [False for _ in range(args.num_processes)]

    reward_stats_deque: List[deque] = [deque(maxlen=500) for _ in range(args.num_processes)]

    num_steps = int(args.T_max // args.num_processes)

    epsilon_start = 1.0
    epsilon_final = args.end_eps
    epsilon_decay = args.eps_decay_period

    def epsilon(t):
        return epsilon_final + (epsilon_start - epsilon_final) * np.exp(
            -1.0 * (t - args.start_timesteps) / epsilon_decay
        )

    start_time = time.time()
    curr_index = 0

    #### Log uniform parameters ####
    loguniform_decay = args.ucb_c * args.diff_eps_schedule_base ** (
        1 + np.arange(args.num_processes)/(args.num_processes-1) * args.diff_eps_schedule_exp)
    loguniform_decay = torch.from_numpy(loguniform_decay).to(args.device).unsqueeze(1)

    #### epsilon-z parameters ####
    n = np.zeros(args.num_processes)
    omega = np.zeros(args.num_processes)
    ez_prob = 1 / np.arange(1, args.eps_z_n+1)**args.eps_z_mu
    ez_prob /= np.sum(ez_prob)
    ez_n = np.arange(1, args.eps_z_n+1)

    for t in range(num_steps):
        if t < args.start_timesteps:
            action = (
                torch.LongTensor([envs.action_space.sample() for _ in range(args.num_processes)])
                .reshape(-1, 1)
                .to(args.device)
            )
            value = agent.get_value(state)
        elif args.explore_strat == "qrdqn_ucb":
            _, mean, var, upper_var = agent.get_quantile(state)
            decay = args.ucb_c * np.sqrt(np.log(t+1) / (t+1))
            value = mean + decay * var
            # print(value.shape)
            action = value.argmax(1).reshape(-1, 1)
            # print(torch.max(mean, 1))
            if t % 500 == 0:
                stats = {
                    "ucb / facotr": decay,
                    "ucb / mean": torch.max(mean, 1)[0].mean().item(),
                    "ucb / upper var": torch.max(upper_var, 1)[0].mean().item(),
                    "ucb / var":  torch.max(var, 1)[0].mean().item(),
                    "ucb / value": torch.max(value, 1)[0].mean().item()
                }
                # print(stats)
                wandb.log(stats, step=t * args.num_processes)
        elif args.qrdqn and args.qrdqn_bootstrap and not args.bootstrap_dqn:
            decay = args.ucb_c
            with torch.no_grad():
                mean, eps_var, ale_var = agent.get_bootstrapped_uncertainty(state)
            total_var = torch.sqrt(eps_var + ale_var)
            eps_var, ale_var = torch.sqrt(eps_var), torch.sqrt(ale_var)
            mean = mean.mean(axis=1)
            if args.thompson_sampling:
                eps_var = eps_var * torch.randn(eps_var.shape, device=eps_var.device)
            # value = mean + decay * eps_var * torch.randn(eps_var.shape, device=eps_var.device)
            # value = mean + decay * eps_var
            if args.diff_epsilon_schedule:
                value = mean + loguniform_decay.expand(args.num_processes, mean.size(1)) * eps_var
            elif args.total_uncertainty:
                value = mean + decay * total_var
            elif args.ale_uncertainty:
                value = mean + decay * ale_var
            else:
                value = mean + decay * eps_var
            action = value.argmax(1).reshape(-1, 1)
            if t % 500 == 0:
                stats = {
                    "ucb / mean": torch.max(mean, 1)[0].mean().item(),
                    "ucb / eps uncertainty":  torch.max(eps_var, 1)[0].mean().item(),
                    "ucb / ale uncertainty":  torch.max(ale_var, 1)[0].mean().item(),
                    "ucb / value": torch.max(value, 1)[0].mean().item()
                }
                wandb.log(stats, step=t * args.num_processes)
        elif args.qrdqn and args.qrdqn_bootstrap and args.bootstrap_dqn:
            if t % 30 == 0:
                curr_index = np.random.randint(args.n_ensemble)
            with torch.no_grad():
                all_quantiles = agent.Q.single_quantile(state, curr_index)  # (B, atom, action)
                value = all_quantiles.mean(axis=1)
                action = value.argmax(1).reshape(-1, 1)
            if t % 500 == 0:
                stats = {
                    "current_idx": torch.max(mean, 1)[0].mean().item(),
                    "ucb / value": torch.max(value, 1)[0].mean().item()
                }
                wandb.log(stats, step=t * args.num_processes)
        elif args.bootstrap_dqn_ucb and args.bootstrap_dqn:
            mean, std = agent.get_bootstrap_dqn_values(state)
            decay = args.ucb_c
            value = mean + decay * std
            action = value.argmax(1).reshape(-1, 1)
            if t % 500 == 0:
                stats = {
                    "ucb / factor": decay,
                    "ucb / mean": torch.mean(mean).item(),
                    "ucb / std":  torch.mean(std).item()
                }
                wandb.log(stats, step=t * args.num_processes)
        elif args.bootstrap_dqn:
            for i in range(args.num_processes):
                if len(action_deque[i]) == 0:
                    # print(f'sampling new head for {i}')
                    agent.current_bootstrap_head[i] = np.random.randint(args.n_ensemble)
            action, value = agent.select_action(state)
            cur_epsilon = epsilon(t)
            for i in range(args.num_processes):
                if np.random.uniform() < cur_epsilon:
                    action[i] = torch.LongTensor([envs.action_space.sample()]).to(args.device)
            if t % 500 == 0:
                wandb.log({"Current Epsilon": cur_epsilon}, step=t * args.num_processes)
        elif args.diff_epsilon_schedule:
            cur_epsilon = args.diff_eps_schedule_base ** (1 + np.arange(args.num_processes)/(args.num_processes-1) * args.diff_eps_schedule_exp)
            action, value = agent.select_action(state)
            for i in range(args.num_processes):
                if np.random.uniform() < cur_epsilon[i]:
                    action[i] = torch.LongTensor([envs.action_space.sample()]).to(args.device)
        elif args.eps_z:
            cur_epsilon = epsilon(t)
            action, value = agent.select_action(state)
            for i in range(args.num_processes):
                if n[i] == 0:
                    if np.random.uniform() < cur_epsilon:
                        n[i] = np.random.choice(ez_n, 1, p=ez_prob)
                        omega[i] = envs.action_space.sample()
                        action[i] = torch.LongTensor([omega[i]]).to(args.device)
                else:
                    action[i] = torch.LongTensor([omega[i]]).to(args.device)
                    n[i] = n[i] - 1
        elif args.noisy_layers:
            if t % args.train_freq == 0:
                agent.Q.reset_noise()
            action, value = agent.select_action(state)
        else:
            cur_epsilon = epsilon(t)
            action, value = agent.select_action(state)
            for i in range(args.num_processes):
                if np.random.uniform() < cur_epsilon:
                    action[i] = torch.LongTensor([envs.action_space.sample()]).to(args.device)
            if t % 500 == 0:
                wandb.log({"Current Epsilon": cur_epsilon}, step=t * args.num_processes)

        # Reset linear layer
        if args.reset_interval > 0 and t % args.reset_interval == 0:
            print(f"Resetting at step {t}")
            for name, module in agent.Q.named_children():
                for keyword in ["linear", "fc"]:
                    if keyword in name:
                        init_(module)
            agent.Q_target = copy.deepcopy(agent.Q)

        if t % 500 and not args.qrdqn or args.c51:
            advantages = agent.advantage(state, epsilon(t))
            mean_max_advantage = advantages.max(1)[0].mean()
            mean_min_advantage = advantages.min(1)[0].mean()
            wandb.log(
                {
                    "Mean Max Advantage": mean_max_advantage,
                    "Mean Min Advantage": mean_min_advantage,
                },
                step=t * args.num_processes,
            )

        # Perform action and log results
        next_state, reward, done, infos = envs.step(action)
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

        for i, info in enumerate(infos):
            if "bad_transition" in info.keys():
                print("Bad transition")
            if level_sampler:
                if expect_new_seed[i]:
                    level_seed = info["level_seed"]
                    level_seeds[i][0] = level_seed
                    if args.log_per_seed_stats:
                        new_episode(value, estimates, level_seed, i, step=t * args.num_processes)
                    expect_new_seed[i] = False
            state_deque[i].append(state[i])
            reward_deque[i].append(reward[i])
            action_deque[i].append(action[i])
            reward_stats_deque[i].append(reward[i])
            if len(state_deque[i]) == args.multi_step or done[i]:
                temp_reward = reward_deque[i]
                # if args.reward_clip > 0:
                #     temp_reward = np.clip(temp_reward, -args.reward_clip, args.reward_clip)
                n_reward = multi_step_reward(temp_reward, args.gamma)
                n_state = state_deque[i][0]
                n_action = action_deque[i][0]
                replay_buffer.add(
                    n_state,
                    n_action,
                    next_state[i],
                    n_reward,
                    np.uint8(done[i]),
                    level_seeds[i],
                )
                if done[i]:
                    reward_deque_i = list(reward_deque[i])
                    for j in range(1, len(reward_deque_i)):
                        n_reward = multi_step_reward(reward_deque_i[j:], args.gamma)
                        n_state = state_deque[i][j]
                        n_action = action_deque[i][j]
                        replay_buffer.add(
                            n_state,
                            n_action,
                            next_state[i],
                            n_reward,
                            np.uint8(done[i]),
                            level_seeds[i],
                        )
                    expect_new_seed[i] = True
            if "episode" in info.keys():
                episode_reward = info["episode"]["r"]
                ppo_normalised_reward = ppo_normalise_reward(episode_reward, args.env_name)
                min_max_normalised_reward = min_max_normalise_reward(episode_reward, args.env_name)
                wandb.log(
                    {
                        "Train Episode Returns": episode_reward,
                        "Train Episode Returns (normalised)": ppo_normalised_reward,
                        "Train Episode Returns (ppo normalised)": ppo_normalised_reward,
                        "Train Episode Returns (min-max normalised)": min_max_normalised_reward,
                    },
                    step=t * args.num_processes,
                )
                state_deque[i].clear()
                reward_deque[i].clear()
                action_deque[i].clear()
                if args.log_per_seed_stats:
                    plot_level_returns(
                        level_seeds,
                        returns,
                        estimates,
                        gaps,
                        episode_reward,
                        i,
                        step=t * args.num_processes,
                    )

        if args.autodrq:
            rollouts.insert(next_state, action, value.unsqueeze(1), torch.Tensor(reward), masks, level_seeds)

        state = next_state

        if args.autodrq and (t + 1) % 256 == 0:
            with torch.no_grad():
                obs_id = rollouts.obs[-1]
                next_value = agent.get_value(obs_id).unsqueeze(1).detach()

            rollouts.compute_returns(next_value, args.gamma, args.gae_lambda)
            replay_buffer.update_ucb_values(rollouts)
            rollouts.after_update()

        # Train agent after collecting sufficient data
        if t % args.train_freq == 0 and t >= args.start_timesteps:
            for _ in range(args.opt_step_per_interaction):
                loss, grad_magnitude, weight = agent.train(replay_buffer)
            if t % 500 == 0:
                wandb.log(
                    {"Value Loss": loss, 
                    "Gradient magnitude": grad_magnitude,
                    },
                    step=t * args.num_processes,
                )
                if args.qrdqn and args.qrdqn_bootstrap and not args.PER:
                    wandb.log(
                        {"weights": torch.mean(weight).item()},
                        step=t * args.num_processes,
                    )

        if t % 500 == 0:
            effective_rank = agent.Q.effective_rank()
            wandb.log({"Effective Rank of DQN": effective_rank}, step=t * args.num_processes)

        if (t + 1) % int((num_steps - 1) / 10) == 0:
            if args.track_seed_weights and not args.PER:
                count_data = [
                    [seed, count]
                    for (seed, count) in zip(agent.seed_weights.keys(), agent.seed_weights.values())
                ]
                total_weight = sum(agent.seed_weights.values())
                count_data = [[i[0], i[1] / total_weight] for i in count_data]
                table = wandb.Table(data=count_data, columns=["Seed", "Weight"])
                wandb.log(
                    {
                        f"Seed Sampling Distribution at time {t}": wandb.plot.bar(
                            table, "Seed", "Weight", title="Sampling distribution of levels"
                        )
                    }
                )
                correlation1 = np.corrcoef(gaps, list(agent.seed_weights.values()))[0][1]
                correlation2 = np.corrcoef(returns, list(agent.seed_weights.values()))[0][1]
                wandb.log(
                    {
                        "Correlation between value error and number of samples": correlation1,
                        "Correlation between empirical return and number of samples": correlation2,
                    }
                )
            else:
                seed2weight = replay_buffer.weights_per_seed()
                weight_data = [
                    [seed, weight] for (seed, weight) in zip(seed2weight.keys(), seed2weight.values())
                ]
                correlation1 = np.corrcoef(gaps, list(seed2weight.values()))[0][1]
                correlation2 = np.corrcoef(returns, list(seed2weight.values()))[0][1]

        if t >= args.start_timesteps and t % args.eval_freq == 0:
            if args.record_td_error:
                test_reward, train_td_loss = eval_policy(args, agent, args.num_test_seeds, print_score=False)
                train_reward, test_td_loss = eval_policy(args,
                                                         agent,
                                                         args.num_test_seeds,
                                                         start_level=0,
                                                         num_levels=args.num_train_seeds,
                                                         seeds=seeds,
                                                         print_score=False)
                mean_test_rewards = np.mean(test_reward)
                mean_train_rewards = np.mean(train_reward)
                wandb.log(
                    {
                        "Train / td loss": train_td_loss,
                        "Test / td loss": test_td_loss,
                        "td loss difference": test_td_loss - train_td_loss
                    },
                    step=t * args.num_processes
                )
            else:
                mean_test_rewards = np.mean(eval_policy(args, agent, args.num_test_seeds, print_score=False))
                mean_train_rewards = np.mean(
                    eval_policy(
                        args,
                        agent,
                        args.num_test_seeds,
                        start_level=0,
                        num_levels=args.num_train_seeds,
                        seeds=seeds,
                        print_score=False
                    )
                )

            if args.advanced_test:
                advanced_stats = {}
                all_exp_c = np.arange(args.exploration_coeff_n) * args.exploration_coeff_mult
                for exp_c in all_exp_c:
                    test_reward, train_td_loss = eval_policy(
                        args, agent, args.num_test_seeds, print_score=False,
                        advanced_test=True, exploration_coeff=exp_c)
                    advanced_stats['Advanced explore / {}'.format(exp_c)] = np.mean(test_reward)
                wandb.log(advanced_stats, step=t * args.num_processes)

            test_ppo_normalised_reward = ppo_normalise_reward(mean_test_rewards, args.env_name)
            train_ppo_normalised_reward = ppo_normalise_reward(mean_train_rewards, args.env_name)
            test_min_max_normalised_reward = min_max_normalise_reward(mean_test_rewards, args.env_name)
            train_min_max_normalised_reward = min_max_normalise_reward(mean_train_rewards, args.env_name)
            if t % 1 == 0:
                all_reward = np.array(reward_stats_deque).reshape(-1)
                all_reward = [[r] for r in all_reward]
                # table = wandb.Table(data=all_reward, columns=["reward"])
                reward_mean = np.mean(all_reward)
                reward_std = np.std(all_reward)
                reward_skew = skew(all_reward)
                reward_kurtosis = kurtosis(all_reward)
                wandb.log(
                    {
                        "Test / Evaluation Returns": mean_test_rewards,
                        "Train / Evaluation Returns": mean_train_rewards,
                        "Generalization Gap:": mean_train_rewards - mean_test_rewards,
                        "Test / Evaluation Returns (normalised)": test_ppo_normalised_reward,
                        "Train / Evaluation Returns (normalised)": train_ppo_normalised_reward,
                        "Test / Evaluation Returns (ppo normalised)": test_ppo_normalised_reward,
                        "Train / Evaluation Returns (ppo normalised)": train_ppo_normalised_reward,
                        "Test / Evaluation Returns (min-max normalised)": test_min_max_normalised_reward,
                        "Train / Evaluation Returns (min-max normalised)": train_min_max_normalised_reward,
                        "Time per step": (time.time() - start_time) / t,
                        "Reward / mean": reward_mean,
                        "Reward / std": reward_std,
                        "Reward / skew": reward_skew,
                        "Reward / kurtosis": reward_kurtosis,
                    },
                    step=t * args.num_processes
                )

                logger.logkv("minutes elapse", (time.time() - start_time) / 60)
                logger.logkv("time / step", (time.time() - start_time) / t)
                logger.logkv("train / total_num_steps", t * args.num_processes)
                logger.logkv("train / mean_episode_reward", mean_train_rewards)
                logger.logkv("test / mean_episode_reward", mean_test_rewards)
                # logger.logkv("train/median_episode_reward", np.median(episode_rewards))
                # logger.logkv("train/loss", np.mean(episode_loss))
                # logger.logkv("test / average_reward", mean_test_rewards)
                # logger.logkv("test/median_reward", np.median(rewards))
                # logger.logkv("test/average_q_values", avg_Q)
                # logger.logkv("time/epsilon", scheduler.get_value(T))
                logger.dumpkvs()

        # if t % 1000 == 0 and t > 1:
        #     hist = np.array(reward_stats_deque).reshape(-1)
        #     hist = np.histogram(hist, density=True)
        #     hist = wandb.Histogram(np_histogram=hist, num_bins=100)
        #     wandb.log({'Reward / reward_histogram': hist}, step=t * args.num_processes)

    print(f"\nLast update: Evaluating on {args.final_num_test_seeds} test levels...\n  ")
    final_eval_episode_rewards = eval_policy(
        args, agent, args.final_num_test_seeds, record=args.record_final_eval
    )

    mean_final_eval_episode_rewards = np.mean(final_eval_episode_rewards)
    median_final_eval_episide_rewards = np.median(final_eval_episode_rewards)

    print("Mean Final Evaluation Rewards: ", mean_final_eval_episode_rewards)
    print("Median Final Evaluation Rewards: ", median_final_eval_episide_rewards)

    wandb.log(
        {
            "Mean Final Evaluation Rewards": mean_final_eval_episode_rewards,
            "Median Final Evaluation Rewards": median_final_eval_episide_rewards,
            "Mean Final Evaluation Rewards (normalised)": ppo_normalise_reward(
                mean_final_eval_episode_rewards, args.env_name
            ),
            "Median Final Evaluation Rewards (normalised)": ppo_normalise_reward(
                median_final_eval_episide_rewards, args.env_name
            ),
        }
    )

    if args.save_model:
        print(f"Saving model to {args.model_path}")
        if "models" not in os.listdir():
            os.mkdir("models")
        torch.save(
            {
                "model_state_dict": agent.Q.state_dict(),
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


def eval_policy(
    args,
    policy,
    num_episodes,
    num_processes=1,
    deterministic=False,
    start_level=0,
    num_levels=0,
    seeds=None,
    level_sampler=None,
    progressbar=None,
    record=False,
    print_score=True,
    advanced_test=False,
    exploration_coeff=1.0
):
    if level_sampler:
        start_level = level_sampler.seed_range()[0]
        num_levels = 1

    eval_envs, level_sampler = make_dqn_lr_venv(
        num_envs=num_processes,
        env_name=args.env_name,
        seeds=seeds,
        device=args.device,
        num_levels=num_levels,
        start_level=start_level,
        no_ret_normalization=args.no_ret_normalization,
        distribution_mode=args.distribution_mode,
        paint_vel_info=args.paint_vel_info,
        level_sampler=level_sampler,
        record_runs=record,
        attach_task_id=args.attach_task_id
    )
    ############################################################
    if args.record_td_error:
        replay_buffer = make_buffer(args, eval_envs)
        level_seeds = torch.zeros(args.num_processes)
        state_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
        reward_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
        action_deque: List[deque] = [deque(maxlen=args.multi_step) for _ in range(args.num_processes)]
        expect_new_seed: List[bool] = [False for _ in range(args.num_processes)]
    ############################################################
    eval_episode_rewards: List[float] = []
    if level_sampler:
        state, _ = eval_envs.reset()
    else:
        state = eval_envs.reset()
    while len(eval_episode_rewards) < num_episodes:
        if advanced_test:
            action = policy.sample_action(state, c=exploration_coeff)
        elif not deterministic and np.random.uniform() < args.eval_eps:
            action = (
                torch.LongTensor([eval_envs.action_space.sample() for _ in range(num_processes)])
                .reshape(-1, 1)
                .to(args.device)
            )
        # if not deterministic:
        #     action, _ = policy.sample_action(state, temperature=0.05)
        else:
            with torch.no_grad():
                action, _ = policy.select_action(state, eval=True)
        next_state, reward, done, infos = eval_envs.step(action)
        ############################################################
        if args.record_td_error:
            for i, info in enumerate(infos):
                if "bad_transition" in info.keys():
                    print("Bad transition")
                state_deque[i].append(state[i])
                reward_deque[i].append(reward[i])
                action_deque[i].append(action[i])
                if len(state_deque[i]) == args.multi_step or done[i]:
                    temp_reward = reward_deque[i]
                    # if args.reward_clip > 0:
                    #     temp_reward = np.clip(temp_reward, -args.reward_clip, args.reward_clip)
                    n_reward = multi_step_reward(temp_reward, args.gamma)
                    n_state = state_deque[i][0]
                    n_action = action_deque[i][0]
                    replay_buffer.add(
                        n_state,
                        n_action,
                        next_state[i],
                        n_reward,
                        np.uint8(done[i]),
                        level_seeds[i],
                    )
                    if done[i]:
                        reward_deque_i = list(reward_deque[i])
                        for j in range(1, len(reward_deque_i)):
                            n_reward = multi_step_reward(reward_deque_i[j:], args.gamma)
                            n_state = state_deque[i][j]
                            n_action = action_deque[i][j]
                            replay_buffer.add(
                                n_state,
                                n_action,
                                next_state[i],
                                n_reward,
                                np.uint8(done[i]),
                                level_seeds[i],
                            )
                        expect_new_seed[i] = True
        ############################################################
        state = next_state
        for info in infos:
            if "episode" in info.keys():
                eval_episode_rewards.append(info["episode"]["r"])
                if progressbar:
                    progressbar.update(1)

    if record:
        for video in eval_envs.get_videos():
            wandb.log({"evaluation_behaviour": video})

    eval_envs.close()
    if progressbar:
        progressbar.close()

    avg_reward = sum(eval_episode_rewards) / len(eval_episode_rewards)

    if print_score:
        print("---------------------------------------")
        print(f"Evaluation over {num_episodes} episodes: {avg_reward}")
        print("---------------------------------------")

    ############################################################
    if args.record_td_error:
        with torch.no_grad():
            n_batch = 2
            loss = 0
            for _ in range(n_batch):
                _, batch_loss, _ = policy.loss(replay_buffer)
                loss += batch_loss.item()
            loss /= n_batch * args.batch_size
        del replay_buffer
        return eval_episode_rewards, loss
    ############################################################
    return eval_episode_rewards


def multi_step_reward(rewards, gamma):
    ret = 0.0
    for idx, reward in enumerate(rewards):
        ret += reward * (gamma ** idx)
    return ret


def new_episode(value, estimates, level_seed, i, step):
    estimates[level_seed] = value[i].item()
    wandb.log(
        {f"Start State Value Estimate for Level {level_seed}": value[i].item()},
        step=step,
    )


def plot_level_returns(level_seeds, returns, estimates, gaps, episode_reward, i, step):
    seed = level_seeds[i][0].item()
    returns[seed] = episode_reward
    gaps[seed] = episode_reward - estimates[seed]
    wandb.log({f"Empirical Return for Level {seed}": episode_reward}, step=step)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.disable(logging.CRITICAL)

    if args.seed_path:
        train_seeds = load_seeds(args.seed_path)
    else:
        train_seeds = generate_seeds(args.num_train_seeds, args.base_seed)

    train(args, train_seeds)
