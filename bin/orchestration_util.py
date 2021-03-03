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

def initialize_tpus(num_tpus, dry_run):
  dry_run_string = "--dry-run" if dry_run else ""
  # start the tpus
  print("Creating %d tpus..." % num_tpus)
  os.system("seq 1 %d |"
            " parallel %s gcloud alpha compute tpus tpu-vm create l2i_tpu_{}" 
            " --zone europe-west4-a --accelerator-type v3-8"
            " --version v2-alpha" % (num_tpus, dry_run_string))

  allow_ssh(dry_run)
  if not dry_run:
    wait_till_tpus_up(num_tpus)

  allow_ssh(dry_run)
  print("Preparing VMs...")
  os.system("seq 1 %d |"
            " parallel %s 'gcloud alpha compute tpus tpu-vm ssh l2i_tpu_{} --zone europe-west4-a"
            " -- \"git clone https://github.com/dieterichlawson/learn_to_infer.git"
            " && pip3 install -r learn_to_infer/requirements.txt\"'" % (num_tpus, dry_run_string))

def wait_till_tpus_up(num_tpus, max_retries=3):

  def check_tpu(tpu_num):
    cmd = ("gcloud alpha compute tpus tpu-vm ssh l2i_tpu_%d"
           " --zone europe-west4-a --command=exit" % tpu_num)
    up = False
    num_tries = 0
    while not up and num_tries < max_retries:
      result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
          shell=True, executable="/bin/zsh")
      up = (result.returncode == 0)
      num_tries += 1
    return up

  pool = ThreadPool(multiprocessing.cpu_count())
  results = []
  for i in range(1, num_tpus + 1):
    results.append(pool.apply_async(check_tpu, (i,)))

  # Close the pool and wait for each running task to complete
  pool.close()
  pool.join()
  results = [r.get() for r in results]
  if all(results):
    print("All TPUs up")
  else:
    down_tpus = [str(i+1) for i,b in enumerate(results) if not b]
    print("TPUs %s failed to come up after %d retries." % (", ".join(down_tpus), max_retries))

def run_commands(commands, dry_run):
  dry_run_string = "--dry-run" if dry_run else ""
  tmux_cmd = "tmux new-session -d \"%s; read;\""
  tmux_cmds = [tmux_cmd % c for c in commands]
  tmux_cmd_string = "\n".join(tmux_cmds)
  allow_ssh(dry_run)
  os.system("echo '%s' |"
            " parallel %s gcloud alpha compute tpus tpu-vm ssh l2i_tpu_{#}"
            " --zone europe-west4-a -- {}" % (tmux_cmd_string, dry_run_string))
