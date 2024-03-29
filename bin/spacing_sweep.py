experiment_name = "spacing_exp"

command = "python3 learn_to_infer/run_gmm.py"

hparams = {
  "model_name": "mean_scale_weight",
  "data_dim": [2, 8, 16],
  "num_encoders": 2,
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": [2, 4, 8, 16],
  "data_points_per_mode": 50,
  "cov_prior": "inv_wishart",
  "dist_multiplier": [.68, .95, .99],
  "batch_size": 64,
  "eval_batch_size": 256,
  "lr": 1e-3,
  "checkpoint_every": 2500,
  "summarize_every": 2500,
  "num_steps": int(1e8),
  "logdir": "gs://l2i/%s" % experiment_name
}
