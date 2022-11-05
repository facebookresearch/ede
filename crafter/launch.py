from ast import parse
import datetime
from distutils import cmd
import submitit
import os
import sys
# from coolname import generate_slug

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("path", "./slurm_tmp/cmds.txt", "Path to list of commands to run.")
flags.DEFINE_string("name", "launch", "Experiment name.")
flags.DEFINE_boolean("debug", False, "Only debugging output.")
flags.DEFINE_string("partition", "learnfair", "partition name.")

def main(argv):
    now = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S-%f")

    rootdir = os.path.expanduser(f"./slurm_logs/{FLAGS.name}")
    submitit_dir = os.path.expanduser(f"./slurm_logs/{FLAGS.name}/{now}")
    executor = submitit.SlurmExecutor(folder=submitit_dir, max_num_timeout=10)
    os.makedirs(submitit_dir, exist_ok=True)

    with open(os.path.expanduser(FLAGS.path), "r") as f:
        cmds = "".join(f.readlines()).split("\n\n")
        cmds = [cmd for cmd in cmds if len(cmd) > 0]
    
    parsed_cmds = []
    for c in cmds:
        print(c)
        parsed_c = []
        for x in c.split(' '):
            if '=' in x:
                for y in x.split('='):
                    parsed_c.append(y)
            else:
                parsed_c.append(x)
        # print(parsed_c)
        parsed_cmds.append(parsed_c)

    # cmd = ["python", "train_rainbow.py"]
    # cmd2 = ["python", "train_rainbow.py", "--qrdqn", "True"]
    # parsed_cmds = [cmd, cmd2]

    executor.update_parameters(
        # examples setup
        partition=FLAGS.partition,
        time=1 * 48 * 60,
        ntasks_per_node=1,
        # job setup
        job_name=FLAGS.name,
        mem="48GB",
        cpus_per_task=16,
        gpus_per_node=1,
        array_parallelism=200,
        # exclude="learnfair026,learnfair123"
    )

    if not FLAGS.debug:
        with executor.batch():
            for c in parsed_cmds:
                print(c)
                function = submitit.helpers.CommandFunction(c)
                job = executor.submit(function)


if __name__ == "__main__":
    app.run(main)
