import orchestration as orch
import argparse

parser = argparse.ArgumentParser(description='Run L2I commands.')

parser.add_argument('--create_and_init_tpus', action="store_true",
                    help='Start new TPUs for these commands')
parser.add_argument('--init_tpus', action="store_true",
                    help='Initialize TPUs for these commands')
parser.add_argument('--dry_run', action="store_true",
                    help='Do a dry run printing commands.')

args = parser.parse_args()

experiment_name = "spacing_exp_2"

command = "python3 learn_to_infer/run_gmm.py"

job_args = {
  "model_name": "mean_scale_weight",
  "data_dim": 2,
  "num_encoders": 2,
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": [2, 3],
  "data_points_per_mode": 50,
  "cov_prior": "inv_wishart",
  "dist_multiplier": [0.10, 0.30, .50, .75, .99],
  "batch_size": 128,
  "eval_batch_size": 256,
  "lr": 1e-3,
  "checkpoint_every": 10000,
  "summarize_every": 2500,
  "num_steps": int(1e8),
  "logdir": "gs://l2i/%s" % experiment_name
}

commands = orch.make_commands(command, job_args)

if args.create_and_init_tpus:
  orch.create_and_init_tpus(experiment_name, len(commands), args.dry_run)
elif args.init_tpus:
  if orch.ensure_tpus_up(experiment_name, len(commands), args.dry_run):
    orch.initialize_tpus(experiment_name, len(commands), args.dry_run)

orch.run_commands(experiment_name, commands, args.dry_run)
