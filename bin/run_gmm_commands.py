import itertools
import os
import argparse

parser = argparse.ArgumentParser(description='Run L2I commands.')

parser.add_argument('--init_tpus', type=bool, default=True, 
                    help='If true, start new TPUs for these commands') 
parser.add_argument('--dry_run', type=bool, default=True,
                    help='If true, do a dry run printing commands.')

args = parser.parse_args()

dry_run_string = "--dry-run" if args.dry_run else ""

def run_shell_cmd(cmd):
  if args.dry_run:
    print(cmd)
  else:
    os.system(cmd)

def make_commands(command, args):
  wrapped_args = dict([(k,v) if type(v) is list else (k,[v]) for k,v in args.items()])
  param_string = " ".join(["--%s={%s}" % (k, k) for k in args.keys()])
  command = command + " " + param_string
  arg_cross = itertools.product(*wrapped_args.values())
  commands = []
  for arg_set in arg_cross:
    arg_dict = dict(zip(args.keys(), arg_set))
    commands.append(command.format(**arg_dict))
  return commands

def allow_ssh():
  print("Allowing ssh login...")
  run_shell_cmd("gcloud compute --project=learning-to-infer firewall-rules create ssh"
            " --direction=INGRESS --priority=1000 --network=default --action=ALLOW"
            " --rules=tcp:22 --source-ranges=0.0.0.0/0")

def initialize_tpus(num_tpus):
  # start the tpus
  print("Creating %d tpus..." % num_tpus)
  os.system("seq 1 %d | parallel %s gcloud alpha compute tpus tpu-vm create l2i_tpu_{}" 
          " --zone europe-west4-a --accelerator-type v3-8 --version v2-alpha" % (num_tpus, dry_run_string))
  allow_ssh()
  print("Preparing VMs...")
  os.system("seq 1 %d | parallel %s 'gcloud alpha compute tpus tpu-vm ssh l2i_tpu_{}"
          " --zone europe-west4-a" 
          " -- \"git clone https://github.com/dieterichlawson/learn_to_infer.git"
          " && pip3 install -r learn_to_infer/requirements.txt\"'" % (num_tpus, dry_run_string))

def run_commands(commands):
  tmux_cmd = "tmux new-session -d \"%s; read;\""
  tmux_cmds = [tmux_cmd % c for c in commands]
  tmux_cmd_string = "\n".join(tmux_cmds)
  allow_ssh()
  os.system("echo '%s' | parallel %s gcloud alpha compute tpus tpu-vm ssh l2i_tpu_{#}"
          " --zone europe-west4-a -- {}" % (tmux_cmd_string, dry_run_string))
  
command = "python3 learn_to_infer/run_gmm.py"

job_args = {
  "model_name": "mean_scale_weight",
  "data_dim": 2,
  "num_encoders": 2,
  "num_decoders": 2,
  "num_heads": 8,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": [2, 3, 4],
  "data_points_per_mode": 50,
  "standardize": [True, False],
  "cov_prior": ["wishart", "inv_wishart"],
  "cov_dof": [4, 10, 25],
  "separation_multiplier": 2.,
  "batch_size": 64,
  "eval_batch_size": 256,
  "lr": 1e-3,
  "checkpoint_every": 2500,
  "summarize_every": 2500,
  "num_steps": int(1e8),
  "logdir": "gs://l2i/data_dim_sweep"
}

commands = make_commands(command, job_args)

if args.init_tpus:
  initialize_tpus(len(commands))

run_commands(commands)
