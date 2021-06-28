experiment_name = "tied_layers_exp"

command = "python3 learn_to_infer/run_gmm.py"

defaults = {
  "model_name": "msw_unconditional",
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 64,
  "value_dim_per_head": 32,
  "cov_prior": "inv_wishart",
  "dist_multiplier": .68,
  "dist": "l2",
  "batch_size": 128,
  "eval_batch_size": 4,
  "lr": 1e-1,
  "summarize_every": 5000,
  "checkpoint_every": 10000,
  "expensive_summarize_every": 10000,
  "num_steps": int(1e9),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}

hparams = [
  {"data_dim": 2, "cov_dof": 4, "data_points_per_mode": 50, "k": [4, 8, 16], "tie_layer_weights":False, "num_encoders": 6},
  {"data_dim": 2, "cov_dof": 4, "data_points_per_mode": 50, "k": [4, 8, 16], "tie_layer_weights":True, "num_encoders": [6, 12, 18]},

  {"data_dim": 4, "cov_dof": 6, "data_points_per_mode": 88, "k": [4, 8, 16], "tie_layer_weights":False, "num_encoders": 6},
  {"data_dim": 4, "cov_dof": 6, "data_points_per_mode": 88, "k": [4, 8, 16], "tie_layer_weights":True, "num_encoders": [6, 12, 18]},

  {"data_dim": 8, "cov_dof": 10, "data_points_per_mode": 155, "k": [4, 8, 16], "tie_layer_weights":False, "num_encoders": 6},
  {"data_dim": 8, "cov_dof": 10, "data_points_per_mode": 155, "k": [4, 8, 16], "tie_layer_weights":True, "num_encoders": [6, 12, 18]},

  {"data_dim": 16, "cov_dof": 18, "data_points_per_mode": 278, "k": [4, 8], "tie_layer_weights":False, "num_encoders": 6},
  {"data_dim": 16, "cov_dof": 18, "data_points_per_mode": 278, "k": [4, 8], "tie_layer_weights":True, "num_encoders": [6, 12, 18]},
]
