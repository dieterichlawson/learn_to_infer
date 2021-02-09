#!/bin/bash

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=16 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=50 \
#  --cov_dof=4 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/data_dim_sweep 

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=16 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=16 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=50 \
#  --cov_dof=18 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/data_dim_sweep 

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=32 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=16 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=50 \
#  --cov_dof=34 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/data_dim_sweep 


python3 run_gmm.py \
  --model_name="mean_scale_weight" \
  --data_dim=128 \
  --num_encoders=2 \
  --num_decoders=2 \
  --num_heads=16 \
  --key_dim=32 \
  --value_dim_per_head=64 \
  --min_k=3 \
  --max_k=3 \
  --data_points_per_mode=50 \
  --cov_dof=34 \
  --separation_multiplier=2. \
  --batch_size=64 \
  --eval_batch_size=256 \
  --lr=1e-3 \
  --checkpoint_every=2500 \
  --summarize_every=2500 \
  --num_steps=100000000 \
  --logdir=gs://l2i/data_dim_sweep 
