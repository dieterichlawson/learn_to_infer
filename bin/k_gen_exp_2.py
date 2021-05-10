experiment_name = "k_gen_exp_2"

command = "python3 learn_to_infer/run_gmm.py"

defaults = {
  "model_name": "mean_scale_weight",
  "num_encoders": 6,
  "num_decoders": 4,
  "num_heads": 16,
  "min_k": 2,
  "max_k": 8,
  "key_dim": 64,
  "value_dim_per_head": 32,
  "cov_prior": "inv_wishart",
  "dist_multiplier": .68,
  "dist": "l2",
  "batch_size": 128,
  "eval_batch_size": 64,
  "summarize_every": 2500,
  "checkpoint_every": 10000,
  "expensive_summarize_every": 10000,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}

hparams = [
 { "data_dim": 2, "data_points_per_mode": 50, "lr": [0.5, 1.] },
 { "data_dim": 4, "data_points_per_mode": 88, "lr": [0.5, 1.] },
 { "data_dim": 8, "data_points_per_mode": 155, "lr": [0.5, 1.] },
]
