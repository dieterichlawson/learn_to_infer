experiment_name = "deep_exp"

command = "python3 learn_to_infer/run_gmm.py"

hparams = {
  "model_name": "msw_unconditional",
  "num_encoders": [12, 16, 20, 24],
  "num_decoders": 2,
  "num_heads": 16,
  "min_k": 8,
  "max_k": 8,
  "data_dim": 8,
  "data_points_per_mode": 155,
  "key_dim": 64,
  "value_dim_per_head": 32,
  "cov_prior": "inv_wishart",
  "dist_multiplier": .68,
  "dist": "l2",
  "batch_size": 128,
  "eval_batch_size": 64,
  "lr": 1e-1,
  "summarize_every": 2500,
  "checkpoint_every": 10000,
  "expensive_summarize_every": 10000,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}
