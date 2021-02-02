#!/bin/bash

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=16 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=2 \
#  --max_k=2 \
#  --data_points_per_mode=50 \
#  --cov_dof=10 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=1000

python3 run_gmm.py \
  --model_name="mean_scale_weight" \
  --data_dim=16 \
  --num_encoders=2 \
  --num_decoders=2 \
  --num_heads=16 \
  --key_dim=32 \
  --value_dim_per_head=64 \
  --min_k=2 \
  --max_k=2 \
  --data_points_per_mode=50 \
  --cov_dof=10 \
  --separation_multiplier=2. \
  --batch_size=64 \
  --eval_batch_size=256 \
  --lr=1e-3 \
  --checkpoint_every=1000
