import itertools
import os

def run_shell_cmd(cmd, dry_run):
  if dry_run:
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

def allow_ssh(dry_run):
  print("Allowing ssh login...")
  run_shell_cmd("gcloud compute --project=learning-to-infer firewall-rules create ssh"
            " --direction=INGRESS --priority=1000 --network=default --action=ALLOW"
            " --rules=tcp:22 --source-ranges=0.0.0.0/0", dry_run)

def initialize_tpus(num_tpus, dry_run):
  dry_run_string = "--dry-run" if dry_run else ""
  # start the tpus
  print("Creating %d tpus..." % num_tpus)
  os.system("seq 1 %d | parallel %s gcloud alpha compute tpus tpu-vm create l2i_tpu_{}" 
          " --zone europe-west4-a --accelerator-type v3-8 --version v2-alpha" % (num_tpus, dry_run_string))
  allow_ssh(dry_run)
  print("Preparing VMs...")
  os.system("seq 1 %d | parallel %s 'gcloud alpha compute tpus tpu-vm ssh l2i_tpu_{}"
          " --zone europe-west4-a" 
          " -- \"git clone https://github.com/dieterichlawson/learn_to_infer.git"
          " && pip3 install -r learn_to_infer/requirements.txt\"'" % (num_tpus, dry_run_string))

def run_commands(commands, dry_run):
  dry_run_string = "--dry-run" if dry_run else ""
  tmux_cmd = "tmux new-session -d \"%s; read;\""
  tmux_cmds = [tmux_cmd % c for c in commands]
  tmux_cmd_string = "\n".join(tmux_cmds)
  allow_ssh(dry_run)
  os.system("echo '%s' | parallel %s gcloud alpha compute tpus tpu-vm ssh l2i_tpu_{#}"
          " --zone europe-west4-a -- {}" % (tmux_cmd_string, dry_run_string))
