import itertools
import os
import multiprocessing
import subprocess
import shlex

from multiprocessing.pool import ThreadPool

def make_commands(command, *hyperparam_groups, defaults={}):
  hyperparam_dicts = make_hyperparam_dicts(*hyperparam_groups, defaults=defaults)
  param_strings = [" ".join(["--%s={%s}" % (k,k) for k in x.keys()]) for x in hyperparam_dicts]
  unformatted_commands = [command + " " + p for p in param_strings]
  commands = [c.format(**hps) for c, hps in zip(unformatted_commands, hyperparam_dicts)]
  return commands

def make_hyperparam_dicts(*hyperparam_groups, defaults={}):
  # take the cross product of each group of hyperparams
  cross_prod_list = [dict_cross_product(d) for d in hyperparam_groups]
  # flatten the list of lists of hyperparameters
  hyperparam_list = itertools.chain.from_iterable(cross_prod_list)
  # add the defaults
  return [{**defaults, **x} for x in hyperparam_list]

def dict_cross_product(d):
  wrapped = dict([(k,v) if type(v) is list else (k,[v]) for k,v in d.items()])
  dict_cross_vals = itertools.product(*wrapped.values())
  dict_list = [dict(zip(d.keys(), x)) for x in dict_cross_vals]
  return dict_list

def run_shell_cmd(cmd, dry_run):
  if dry_run:
    print(cmd)
  else:
    os.system(cmd)

def allow_ssh(dry_run):
  print("Allowing ssh login...")
  run_shell_cmd("gcloud compute --project=learning-to-infer firewall-rules create ssh"
                " --direction=INGRESS --priority=1000 --network=default --action=ALLOW"
                " --rules=tcp:22 --source-ranges=0.0.0.0/0", dry_run)

def create_and_init_tpus(basename, num_tpus, dry_run):
  create_tpus(basename, num_tpus, dry_run)
  if ensure_tpus_up(basename, num_tpus, dry_run):
    initialize_tpus(basename, num_tpus, dry_run)
  else:
    print("Not initializing TPUs")

def create_tpus(basename, num_tpus, dry_run):
  dry_run_string = "--dry-run" if dry_run else ""
  # start the tpus
  print("Creating %d tpus..." % num_tpus)
  os.system("seq 1 %d |"
            " parallel %s gcloud alpha compute tpus tpu-vm create l2i_%s_{}" 
            " --zone europe-west4-a --accelerator-type v3-8"
            " --version v2-alpha" % (num_tpus, dry_run_string, basename))

def ensure_tpus_up(basename, num_tpus, dry_run):
  allow_ssh(dry_run)
  if not dry_run:
    return wait_till_tpus_up(basename, num_tpus)
  else:
    return True

def initialize_tpus(basename, num_tpus, dry_run):
  allow_ssh(dry_run)
  dry_run_string = "--dry-run" if dry_run else ""
  print("Preparing VMs...")
  os.system("seq 1 %d |"
            " parallel %s --jobs %d 'gcloud alpha compute tpus tpu-vm ssh l2i_%s_{} --zone europe-west4-a"
            " -- \"git clone https://github.com/dieterichlawson/learn_to_infer.git"
            " && pip3 install -r learn_to_infer/requirements.txt\"'" % (
              num_tpus, dry_run_string, num_tpus, basename))

def reinit_tpus(basename, tpu_nums, dry_run):
  allow_ssh(dry_run)
  dry_run_string = "--dry-run" if dry_run else ""
  tpu_num_string = "\n".join([str(x) for x in tpu_nums])
  print("Resetting TPUs")
  os.system("echo '%s' | parallel %s --jobs %d "
      " 'gcloud alpha compute tpus tpu-vm ssh l2i_%s_{} --zone europe-west4-a"
      " -- \"tmux kill-server"
      " && pip3 uninstall -r learn_to_infer/requirements.txt"
      " && rm -r -f learn_to_infer"
      " && git clone https://github.com/dieterichlawson/learn_to_infer.git"
      " && pip3 install -r learn_to_infer/requirements.txt\"'" % (
        tpu_num_string, dry_run_string, len(tpu_nums), basename))

def wait_till_tpus_up(basename, num_tpus, max_retries=3):

  def check_tpu(tpu_num):
    cmd = ("gcloud alpha compute tpus tpu-vm ssh l2i_%s_%d"
           " --zone europe-west4-a --command=exit" % (basename, tpu_num))
    up = False
    num_tries = 0
    while not up and num_tries < max_retries:
      result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
          shell=True, executable="/bin/zsh")
      up = (result.returncode == 0)
      num_tries += 1
    return up

  print("Waiting for TPUs to come up...")
  pool = ThreadPool(num_tpus)
  results = []
  for i in range(1, num_tpus + 1):
    results.append(pool.apply_async(check_tpu, (i,)))

  # Close the pool and wait for each running task to complete
  pool.close()
  pool.join()
  results = [r.get() for r in results]
  if all(results):
    print("All TPUs up.")
    return True
  else:
    down_tpus = [str(i+1) for i,b in enumerate(results) if not b]
    print("TPUs %s failed to come up after %d retries." % (", ".join(down_tpus), max_retries))
    return False

def run_commands(basename, commands, dry_run):
  run_commands_on_tpus(basename, commands, range(1, len(commands)+1), dry_run)

def run_commands_on_tpus(basename, commands, tpu_nums, dry_run):
  num_tpus = len(commands)
  dry_run_string = "--dry-run" if dry_run else ""
  tmux_cmd = "%d	tmux new-session -d \"%s --tag=%d; read;\""
  tmux_cmds = [tmux_cmd % (i+1, c, i+1) for i, c in zip(tpu_nums, commands)]
  tmux_cmd_string = "\n".join(tmux_cmds)
  allow_ssh(dry_run)
  os.system("echo '%s' |"
            " parallel --colsep '	' --jobs %d %s gcloud alpha compute tpus tpu-vm ssh l2i_%s_{1}"
            " --zone europe-west4-a -- {2}" % (tmux_cmd_string, num_tpus, dry_run_string, basename))
