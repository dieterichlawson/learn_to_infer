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

experiment_name = "layernorm_exp"

command = "python3 learn_to_infer/run_gmm.py"

defaults = {
  "model_name": "mean_scale_weight",
  "data_dim": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "data_points_per_mode": 50,
  "cov_dof": 16,
  "cov_prior": "inv_wishart",
  "dist_multiplier": .50,
  "batch_size": 128,
  "eval_batch_size": 256,
  "lr": 1e-3,
  "checkpoint_every": 10000,
  "summarize_every": 2500,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}

hparams = [
    { "num_encoders": 2,
      "num_decoders": 2,
      "k": [2,3]},
    { "num_encoders": 3,
      "num_decoders": 3,
      "k": [2,3]},
    { "num_encoders": 4,
      "num_decoders": 4,
      "k": [2,3]},
    { "num_encoders": 5,
      "num_decoders": 5,
      "k": [2,3]},
]

commands = orch.make_commands(command, *hparams, defaults=defaults)

if args.create_and_init_tpus:
  orch.create_and_init_tpus(experiment_name, len(commands), args.dry_run)
elif args.init_tpus:
  if orch.ensure_tpus_up(experiment_name, len(commands), args.dry_run):
    orch.initialize_tpus(experiment_name, len(commands), args.dry_run)

orch.run_commands(experiment_name, commands, args.dry_run)
