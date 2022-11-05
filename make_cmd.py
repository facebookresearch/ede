# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import argparse
import json
import os
import itertools
import uuid


def generate_command(params, script_name, newlines=False):
    separator = " \\\n" if newlines else " "
    cmd = []
    base_cmd = f"python -u {script_name}"
    cmd.append(base_cmd)
    for k, v in params.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is False:
            continue
        else:
            cmd.append(f"--{k}={v}")
    cmd = separator.join(cmd)
    return cmd


def parse_args():
    parser = argparse.ArgumentParser(description="Make commands")
    parser.add_argument(
        "--output_dir", type=str, default="slurm_tmp", help="file for storing "
    )
    parser.add_argument(
        "--custom_name",
        type=str,
        default="",
        help="Optional identifier for the experiment",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no_wandb", action="store_true", help="Upload to wandb")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    script_name = "train_rainbow.py"
    hparams = {
        "algo": ["rainbow"],
        # "env_name": ["Crafter"],
        # "env_name": ["bigfish", "plunder", "jumper", "heist", "miner"],
        # "env_name": ["miner", "ninja", "plunder", "starpilot", 
        #              "heist", "jumper", "leaper", "maze", "climber", "coinrun",
        #              "dodgeball", "fruitbot", "bigfish", "bossfight", "caveflyer", "chaser"],
        # "env_name": ["bigfish", "miner", "plunder"],
        # "env_name": ["miner", "ninja", "plunder", "starpilot", 
        #              "heist", "jumper", "leaper", "maze", "climber", "coinrun",
        #              "dodgeball", "fruitbot", "bigfish", "bossfight", "caveflyer", "chaser"],
            # "env_name": ["miner", "ninja", "plunder", 
            #              "leaper", "maze", "climber", "coinrun", "heist",
            #              "fruitbot", "bigfish", "bossfight", "caveflyer", "chaser"],
        "env_name": ["ninja", "plunder", "starpilot", 
                     "jumper", "leaper", "climber",
                     "dodgeball","bigfish", "bossfight", "caveflyer"],
        "qrdqn": ["True"],
        "qrdqn_bootstrap": ["True"],
        # "thompson_sampling": ["True"],
        # "explore_strat": ["qrdqn_ucb"],
        # "reset_interval": [30000, 40000, 50000],
        # "ucb_c": [50, 100, 200],
        # "total_uncertainty": ["False"],
        # "ale_uncertainty": ["True"],
        # "ucb_c": [10, 30, 50],
        # "ucb_c": [0.25, 0.5, 1.0],
        # "ucb_c": [0.2],  # custom for bootstrapping
        "seed": list(range(2)),
        # "uadqn": ["True"],
        # "anchor_loss": [0.01, 0.1],
        # "num_train_seeds": [1],
        # "reward_clip": [0, 10],
        # "multi_step": [5, 7]
        # "memory_capacity": [2e6],
        # "opt_step_per_interaction": [1, 2, 3],
        # "num_processes": [16, 32, 64],
        # "num_processes": [16],
        # "batch_size": [64],
        # "target_update": [16000, 32000],
        # "use_wbb": ["True", "False"],
        # "use_average_target": ["True"],
        # "wbb_temperature": [10, 20],
        # "diff_epsilon_schedule": ["True"],
        # "PER": ["True"],
        # "diff_eps_schedule_exp": [6, 7, 8],
        # "diff_eps_schedule_base": [0.5, 0.6, 0.7],
        # "eps_z": ["True"],
        # "bootstrap_dqn": ["True"],
        # "bootstrap_dqn_ucb": ["True"],
        # "distribution_mode": ["hard"],
        # "num_train_seeds": [1],
        "n_ensemble": [3],
        # "double_qrdqn": ["False"]
        # "drq": ["True", "False"],
        # "drq": ["True"],
        "qrdqn_always_train_feat": ["True", "False"],
        # "noisy_layers": ["True"],
        "record_td_error": ["True"],
        # "advanced_test": ["True"],
        # "exploration_coeff_mult": [.03],
        # "eval_freq": [5000],
        # "attach_task_id": ["True", "False"],
    }

    hparams_names = list(hparams.keys())
    hparams_values = [hparams[k] for k in hparams_names]
    hparams_values = itertools.product(*hparams_values)

    all_cmd = []
    count = 0
    base_log_dir = os.path.join(
        "logs",
        datetime.datetime.now().strftime("%y_%m_%d-%H_%M") + "_" + args.custom_name,
    )
    for p in hparams_values:
        exp_config = dict(zip(hparams_names, p))
        now = datetime.datetime.now().strftime("%y_%m_%d-%H_%M")
        exp_name = "_".join(
            [
                now,
                args.custom_name,
                "_".join(
                    f"{k}_{v}"
                    for k, v in sorted(exp_config.items())
                    if len(hparams[k]) > 1
                ),
            ]
        )
        exp_config["exp_name"] = exp_name
        exp_config["custom_name"] = args.custom_name
        exp_config["log_dir"] = base_log_dir
        if args.wandb:
            exp_config["wandb"] = "True"
        cmd = generate_command(exp_config, script_name)
        all_cmd.append(cmd)
        print(cmd + "\n")
        count += 1

    save_path = os.path.join(
        os.path.expandvars(os.path.expanduser(args.output_dir)), "cmds.txt"
    )
    print(f"Generated {count} commands.")
    print(f"Saving to {save_path}")
    with open(os.path.expanduser(save_path), "w") as f:
        for cmd in all_cmd:
            f.write(cmd + "\n\n")
