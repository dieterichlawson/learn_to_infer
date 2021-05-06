experiment_name = "kl_dist_exp"

command = "python3 learn_to_infer/run_gmm.py"

hparams = {
  "model_name": "msw_unconditional",
  "data_dim": [2, 8],
  "num_encoders": 4,
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": 3,
  "data_points_per_mode": 100,
  "cov_prior": "inv_wishart",
  "dist_multiplier": .68,
  "dist": ["symm_kl", "l2"],
  "batch_size": 128,
  "eval_batch_size": 256,
  "lr": 1e-3,
  "checkpoint_every": 10000,
  "summarize_every": 2500,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}
