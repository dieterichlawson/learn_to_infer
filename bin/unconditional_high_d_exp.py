experiment_name = "unconditional_high_d_exp"

command = "python3 learn_to_infer/run_gmm.py"

job_args = {
  "model_name": "mean_unconditional",
  "data_dim": [16, 32, 64],
  "num_encoders": [4, 6],
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": 2,
  "data_points_per_mode": 128,
  "dist_multiplier": .68,
  "batch_size": 16,
  "eval_batch_size": 64,
  "lr": 1e-3,
  "checkpoint_every": 10000,
  "summarize_every": 2500,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}
