# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser(description="DQN")

# Generic arguments.
parser.add_argument(
    "--exp_name", type=str, default="off-policy", help="name of the experiment"
)
parser.add_argument(
    "--custom_name", type=str, default="", help="Custom name for the experiment"
)
# parser.add_argument("--log_dir", default="logs", help="directory to save agent logs")
parser.add_argument("--save_dir", type=str, default="models", help="augmentation type")
parser.add_argument(
    "--algo", default="dqn", choices=["rainbow", "dqn"], help="algorithm to use"
)
parser.add_argument(
    "--explore_strat", default="", choices=["", "qrdqn_ucb"], help="exploration strat"
)
parser.add_argument(
    "--ucb_c", default=50., type=float, help="coefficient for ucb"
)
parser.add_argument(
    "--reset_interval", default=-1, type=int, help="how often to reset the linear layer"
)
parser.add_argument("--qrdqn_bootstrap", type=lambda x: bool(strtobool(x)), default=False, help="bootstrapped qrdqn style")
parser.add_argument("--qrdqn_always_train_feat", type=lambda x: bool(strtobool(x)), default=False, help="not freeze feature for bootstrapped qrdqn")
parser.add_argument("--double_qrdqn", type=lambda x: bool(strtobool(x)), default=False, help="doubl dqn for qrdqn")
parser.add_argument("--n_ensemble", type=int, default=5, help="Number of ensemble in bootstrap")
parser.add_argument("--use_average_target", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--total_uncertainty", type=lambda x: bool(strtobool(x)), default=False, help="use total uncertainty for exploration")
parser.add_argument("--ale_uncertainty", type=lambda x: bool(strtobool(x)), default=False, help="use aleatoric uncertainty for exploration")
parser.add_argument("--use_wbb", type=lambda x: bool(strtobool(x)), default=False, help="use uncertainty weighted update")
parser.add_argument("--wbb_temperature", type=float, default=10.0, help="Temperature used for wbb")
parser.add_argument("--thompson_sampling", type=lambda x: bool(strtobool(x)), default=False, help="use thompson sampling instead of ucb")
parser.add_argument("--anchor_loss", type=float, default=0.0, help="Multiplier for regression against the prior")
parser.add_argument("--uadqn", type=lambda x: bool(strtobool(x)), default=False, help="use uadqn")

parser.add_argument("--opt_step_per_interaction", type=int, default=1, help="Number of gradient step per interaction")

parser.add_argument("--diff_epsilon_schedule", type=lambda x: bool(strtobool(x)), default=False, help="apex style exploration")
parser.add_argument("--diff_eps_schedule_base", type=float, default=0.6, help="epsilon base for apex style exploration")
parser.add_argument("--diff_eps_schedule_exp", type=float, default=7, help="epsilon base for apex style exploration")

parser.add_argument("--eps_z", type=lambda x: bool(strtobool(x)), default=False, help="Use epsilon z exploration")
parser.add_argument("--eps_z_n", type=int, default=10000, help="n for epsilon z exploration")
parser.add_argument("--eps_z_mu", type=float, default=2.0, help="mu for epsilon z exploration")

parser.add_argument("--bootstrap_dqn", type=lambda x: bool(strtobool(x)), default=False, help="bootstrapped dqn style")
parser.add_argument("--bootstrap_dqn_ucb", type=lambda x: bool(strtobool(x)), default=False, help="ucb for bootstrapped dqn")

parser.add_argument("--attach_task_id", type=lambda x: bool(strtobool(x)), default=False, help="add task id to the state")

parser.add_argument("--record_td_error", type=lambda x: bool(strtobool(x)), default=False, help="record td error for evaluation")
parser.add_argument("--advanced_test", type=lambda x: bool(strtobool(x)), default=False, help="not use e-greedy at test")
parser.add_argument('--exploration_coeff_mult', default=1, type=float)
parser.add_argument('--exploration_coeff_n', default=10, type=int)
###########################################################################
###########################################################################

# Training parameters
parser.add_argument(
    "--start_timesteps", type=int, default=2000, help="Timesteps until using DQN to take actions"
)
parser.add_argument("--train_freq", type=int, default=1, help="Number of steps between DQN updates")
parser.add_argument("--eval_freq", type=int, default=1000, help="Evaluation frequency")
parser.add_argument("--T_max", type=int, default=25e6, help="Total environment steps")
parser.add_argument("--max_episode_length", type=int, default=108e3, help="Max timesteps in one episode")

# Model parameters
parser.add_argument("--c51", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--dueling", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--noisy_layers", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--qrdqn", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--drq", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--autodrq", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--atc", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--simple_dqn", type=lambda x: bool(strtobool(x)), default=False, help="simple dqn arch")
parser.add_argument("--learning_rate", type=float, default=2.5e-4, help="learning rate")
parser.add_argument("--no_cuda", type=lambda x: bool(strtobool(x)), default=False, help="disables gpu")
parser.add_argument("--adam_eps", type=float, default=1.5e-4)
parser.add_argument("--optimizer", default="Adam", help="Optimizer to use")
parser.add_argument(
    "--polyak_target_update",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="Whether to use polyak update to target network",
)
parser.add_argument("--target_update", type=int, default=32000, help="How often to update target network")
parser.add_argument("--tau", type=float, default=0.005, help="tau")
parser.add_argument("--initial_eps", type=float, default=1, help="intial epsilon")
parser.add_argument("--end_eps", type=float, default=0.1, help="end epsilon")
parser.add_argument("--eps_decay_period", type=int, default=8000)
parser.add_argument("--eval_eps", type=float, default=0.05)
parser.add_argument("--min_priority", type=float, default=1e-2)
parser.add_argument("--V_min", type=float, default=0)
parser.add_argument("--V_max", type=float, default=10)
parser.add_argument("--batch_size", type=int, default=512, help="Batch Size")
parser.add_argument("--norm_clip", type=float, default=10)
parser.add_argument("--atoms", type=int, default=51, help="Number of atoms for distributional RL")
parser.add_argument("--hidden_size", type=int, default=512)
parser.add_argument("--noisy_std", type=float, default=0.5, help="Standard deviation for noisy layers")
parser.add_argument("--model", default=None)
parser.add_argument("--history_length", type=int, default=1)
parser.add_argument("--multi_step", type=int, default=3, help="Number of steps for multi step rewards")
parser.add_argument("--t", type=int, default=0)
parser.add_argument("--no_ret_normalization", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--eps", type=float, default=1e-05)
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--disable_checkpoint", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--entropy_coef", type=float, default=0.01)
parser.add_argument("--final_num_test_seeds", type=int, default=1000)
parser.add_argument("--full_train_distribution", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--PER", type=lambda x: bool(strtobool(x)), default=False, help="Whether to use PER")
parser.add_argument(
    "--rank_based_PER", type=lambda x: bool(strtobool(x)), default=False, help="Whether to use rank based PER"
)
parser.add_argument("--beta", type=float, default=0.4, help="Beta value for PER")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha value for PER")
parser.add_argument("--ERE", type=lambda x: bool(strtobool(x)), default=False, help="Whether to use ERE")
parser.add_argument("--per_seed_buffer", type=lambda x: bool(strtobool(x)), default=False)

# Environment parameters
parser.add_argument("--num_processes", type=int, default=64)
parser.add_argument("--env_name", default="starpilot")
parser.add_argument("--distribution_mode", default="easy")
parser.add_argument("--base_seed", type=int, default=0)
parser.add_argument("--paint_vel_info", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--use_sequential_levels", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--start_level", type=int, default=0)
parser.add_argument("--render", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--checkpoint_interval", type=int, default=0)
parser.add_argument("--memory_capacity", type=float, default=1e6)
parser.add_argument("--reward_clip", type=float, default=0)
parser.add_argument(
    "--arch", type=str, default="large", choices=["small", "large", "simple"], help="agent architecture"
)

# Level Replay parameters
parser.add_argument("--num_steps", type=int, default=256, help="number of steps in rollout for plr scores")
parser.add_argument("--level_replay_alpha", type=float, default=1.0)
parser.add_argument("--level_replay_eps", type=float, default=0.05)
parser.add_argument("--level_replay_nu", type=float, default=0.5)
parser.add_argument("--level_replay_rho", type=float, default=1.0)
parser.add_argument("--level_replay_schedule", default="proportionate")
parser.add_argument("--level_replay_score_transform", default="rank")
parser.add_argument("--level_replay_strategy", default="random")
parser.add_argument("--level_replay_temperature", type=float, default=0.1)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--num_test_seeds", type=int, default=10)
parser.add_argument("--num_train_seeds", type=int, default=200)
parser.add_argument("--ppo_epoch", type=int, default=3)
parser.add_argument("--save_interval", type=int, default=60)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--seed_path", default=None)
parser.add_argument("--staleness_coef", type=float, default=0.1)
parser.add_argument("--staleness_temperature", type=float, default=1.0)
parser.add_argument("--staleness_transform", default="power")
parser.add_argument("--value_loss_coef", type=float, default=0.5)
parser.add_argument("--weight_log_interval", type=int, default=1)

# Logging
parser.add_argument(
    "--track_seed_weights",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="Whether to track seed weights when doing experience replay",
)
parser.add_argument(
    "--wandb",
    type=lambda x: bool(strtobool(x)),
    default=False,
    help="Whether to log with wandb or save results locally",
)
parser.add_argument(
    "--wandb_project",
    type=str,
    default="off-policy-procgen",
    choices=["off-policy-procgen", "thesis-experiments"],
)
parser.add_argument(
    "--wandb_tags",
    type=str,
    default="",
    help="Additional tags for this wandb run",
)
parser.add_argument(
    "--wandb_group",
    type=str,
    default="",
    help="Wandb group for this run",
)
parser.add_argument("--log_per_seed_stats", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--record_final_eval", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--log_dir", default="~/dqn_procgen/logs/")
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--xpid", default="latest")
parser.add_argument("--verbose", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--save_model", type=lambda x: bool(strtobool(x)), default=False)
parser.add_argument("--model_path", default="models/model.tar")
parser.add_argument("--interactive", action="store_true", help="whether to show tqdm loop logging")
