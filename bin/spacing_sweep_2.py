experiment_name = "spacing_exp_2"

command = "python3 learn_to_infer/run_gmm.py"

hparams = {
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
