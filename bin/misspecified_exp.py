experiment_name = "misspecified_exp"

command = "python3 learn_to_infer/run_gmm.py"

job_args = {
  "model_name": "fixed_k",
  "data_dim": 2,
  "num_encoders": [3, 4, 6],
  "num_decoders": [2, 3, 4],
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "min_k": 2,
  "max_k": 5,
  "algo_k": 5,
  "fix_em_k": True,
  "data_points_per_mode": 50,
  "cov_prior": "inv_wishart",
  "cov_dof": 16,
  "dist_multiplier": .68,
  "dist": "l2",
  "batch_size": 128,
  "eval_batch_size": 256,
  "lr": 1e-3,
  "summarize_every": 2500,
  "checkpoint_every": 20000,
  "expensive_summarize_every": 20000,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}
