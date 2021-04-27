import orchestration as orch
import argparse
import importlib

parser = argparse.ArgumentParser(description='Run L2I experiments.')

parser.add_argument('exp', help="Experiment to run.")
parser.add_argument('--create_and_init_tpus', action="store_true",
                    help='Start new TPUs for these commands')
parser.add_argument('--init_tpus', action="store_true",
                    help='Initialize TPUs for these commands')
parser.add_argument('--dry_run', action="store_true",
                    help='Do a dry run printing commands.')
parser.add_argument('--reinit_tpus', type=str,
                    help='Stop the tpu computation, deploy new code, and restart.')

args = parser.parse_args()

exp = importlib.import_module(args.exp)

experiment_name = exp.experiment_name
command = exp.command
job_args = exp.job_args

if args.reinit_tpus is not None:
  args.reinit_tpus = [int(x) for x in args.reinit_tpus.split(",")]

commands = orch.make_commands(command, job_args)

if args.reinit_tpus is not None and len(args.reinit_tpus) > 0:
  filtered_cmds = [commands[i-1] for i in args.reinit_tpus]
  orch.reinit_tpus(experiment_name, args.reinit_tpus, args.dry_run)
  orch.run_commands_on_tpus(experiment_name, filtered_cmds, args.reinit_tpus, args.dry_run)

else:
  if args.create_and_init_tpus:
    orch.create_and_init_tpus(experiment_name, len(commands), args.dry_run)
  elif args.init_tpus:
    if orch.ensure_tpus_up(experiment_name, len(commands), args.dry_run):
      orch.initialize_tpus(experiment_name, len(commands), args.dry_run)

  orch.run_commands(experiment_name, commands, args.dry_run)
