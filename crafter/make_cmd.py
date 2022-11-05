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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    script_name = "main_qrdqn.py"
    # hparams = {
    #     "algorithm": ["qrdqn"],
    #     "seed": list(range(10)),
    #     "explore_strat": ["thompson"],
    #     "bootstrapped_qrdqn": ["True"],
    #     "ucb_c": [0.5, 0.3],
    #     "qrdqn_always_train_feat": ["False"],
    #     "T-max": [int(1.5e6)],
    #     "batch-size": [32, 64]
    # }
    hparams = {
        "algorithm": ["rainbow"],
        "seed": list(range(5)),
        "T-max": [int(1.5e6)],
        "batch-size": [32, 64, 128]
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
    try:
        os.mkdir(base_log_dir)
    except:
        print('{} exists'.format(base_log_dir))

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
        exp_config["id"] = exp_name
        # exp_config["custom_name"] = args.custom_name
        exp_config["logdir"] = os.path.join(base_log_dir, exp_name)
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
