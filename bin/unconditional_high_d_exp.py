experiment_name = "unconditional_high_d_exp_2"

command = "python3 learn_to_infer/run_gmm.py"

hparams = {
  "model_name": "mean_unconditional",
  "data_dim": [16, 32],
  "num_encoders": [6, 8],
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": 2,
  "data_points_per_mode": 128,
  "dist_multiplier": .68,
  "batch_size": 16,
  "eval_batch_size": 64,
  "lr": [0.01, 0.1, 0.5],
  "checkpoint_every": 10000,
  "summarize_every": 2500,
  "expensive_summarize_every": 10000,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}
