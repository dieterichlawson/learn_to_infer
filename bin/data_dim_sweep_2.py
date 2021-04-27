import orchestration as orch
import argparse

parser = argparse.ArgumentParser(description='Run L2I commands.')

parser.add_argument('--create_and_init_tpus', action="store_true",
                    help='Start new TPUs for these commands')
parser.add_argument('--init_tpus', action="store_true",
                    help='Initialize TPUs for these commands')
parser.add_argument('--dry_run', action="store_true",
                    help='Do a dry run printing commands.')
parser.add_argument('--reinit_tpus', type=str,
                    help='Stop the tpu computation, deploy new code, and restart.')


args = parser.parse_args()

if args.reinit_tpus is not None:
  args.reinit_tpus = [int(x) for x in args.reinit_tpus.split(",")]

experiment_name = "data_dim_exp_2"

command = "python3 learn_to_infer/run_gmm.py"

defaults = {
  "model_name": "mean_scale_weight",
  "num_encoders": 4,
  "num_decoders": 4,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": 2,
  "cov_prior": "inv_wishart",
  "dist_multiplier": .68,
  "batch_size": 64,
  "eval_batch_size": 256,
  "lr": 1e-3,
  "checkpoint_every": 10000,
  "summarize_every": 2500,
  "normalization": "layer_norm",
  "num_steps": int(1e8),
  "logdir": "gs://l2i/%s" % experiment_name
}

hparams = [
    {"data_dim": 2, "cov_dof": 8, "data_points_per_mode": 50},
    {"data_dim": 4, "cov_dof": 12, "data_points_per_mode": [160, 80, 40]},
    {"data_dim": 8, "cov_dof": 24, "data_points_per_mode": [600, 300, 150]},
    {"data_dim": 16, "cov_dof": 48, "data_points_per_mode": [2250, 1125, 560],
      "batch_size": 8, "eval_batch_size":16},
]

commands = orch.make_commands(command, *hparams, defaults=defaults)

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
