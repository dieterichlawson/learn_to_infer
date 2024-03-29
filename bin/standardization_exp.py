experiment_name = "standardization_exp"

command = "python3 learn_to_infer/run_gmm.py"

hparams = {
  "model_name": "mean_scale_weight",
  "data_dim": 2,
  "num_encoders": 2,
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 32,
  "value_dim_per_head": 32,
  "k": [2, 3, 4],
  "data_points_per_mode": 50,
  "standardize_data": [True, False],
  "cov_prior": ["wishart", "inv_wishart"],
  "cov_dof": [4, 10, 25],
  "separation_multiplier": 2.,
  "batch_size": 64,
  "eval_batch_size": 256,
  "lr": 1e-3,
  "checkpoint_every": 2500,
  "summarize_every": 2500,
  "num_steps": int(1e8),
  "logdir": "gs://l2i/standardization_sweep"
}
