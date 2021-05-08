experiment_name = "num_data_points_exp"

command = "python3 learn_to_infer/run_gmm.py"

defaults = {
  "model_name": "msw_unconditional",
  "num_encoders": 6,
  "num_decoders": 2,
  "num_heads": 16,
  "key_dim": 64,
  "value_dim_per_head": 32,
  "cov_prior": "inv_wishart",
  "k": 4,
  "dist_multiplier": .68,
  "dist": "l2",
  "batch_size": 128,
  "eval_batch_size": 256,
  "lr": 1e-1,
  "summarize_every": 2500,
  "checkpoint_every": 10000,
  "expensive_summarize_every": 10000,
  "num_steps": int(1e8),
  "normalization": "layer_norm",
  "logdir": "gs://l2i/%s" % experiment_name
}

hparams = [
    {
      "data_points_per_mode":12, 
      "test_data_points_per_mode":"12,50,200", 
      "data_dim": [2,4,8]
    },
    {
      "data_points_per_mode":25, 
      "test_data_points_per_mode":"12,50,200", 
      "data_dim": [2,4,8]
    },
    {
      "data_points_per_mode":50, 
      "test_data_points_per_mode":"12,50,200", 
      "data_dim": [2,4,8]
    },
    {
      "data_points_per_mode":100, 
      "test_data_points_per_mode":"12,50,200", 
      "data_dim": [2,4]
    },
    {
      "data_points_per_mode":100, 
      "test_data_points_per_mode":"12,50,200", 
      "data_dim": 8, 
      "eval_batch_size": 200 
    },
    {
      "data_points_per_mode":200, 
      "test_data_points_per_mode":"12,50,200", 
      "data_dim": [2,4,8],
      "eval_batch_size": 200
    }
]
