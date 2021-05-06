experiment_name = "lr_exp"

command = "python3 learn_to_infer/run_gmm.py"

hparams = {
  "model_name": "msw_unconditional",
  "data_dim": 2,
  "num_encoders": 4,
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": [2, 4],
  "data_points_per_mode": 50,
  "cov_prior": "inv_wishart",
  "cov_dof": 16,
  "dist_multiplier": .68,
  "dist": "l2",
  "batch_size": 128,
  "eval_batch_size": 256,
  "lr": [1e-2, 1e-1, 0.5, 1.0],
  "checkpoint_every": 10000,
  "summarize_every": 2500,
  "expensive_summarize_every": 10000,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}
