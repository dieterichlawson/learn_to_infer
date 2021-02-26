import orchestration_util as orch
import argparse

parser = argparse.ArgumentParser(description='Run L2I commands.')

parser.add_argument('--init_tpus', type=bool, default=True, 
                    help='If true, start new TPUs for these commands') 
parser.add_argument('--dry_run', type=bool, default=True,
                    help='If true, do a dry run printing commands.')

args = parser.parse_args()

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

commands = orch.make_commands(command, job_args)

if args.init_tpus:
  orch.initialize_tpus(len(commands), args.dry_run)

orch.run_commands(commands, args.dry_run)