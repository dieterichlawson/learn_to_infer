#!/bin/bash

python3 run_gmm.py \
  --model_name="mean_scale_weight" \
  --data_dim=2 \
  --cov_dof=4 \
  --num_encoders=2 \
  --num_decoders=2 \
  --num_heads=16 \
  --key_dim=32 \
  --value_dim_per_head=64 \
  --min_k=3 \
  --max_k=3 \
  --data_points_per_mode=25 \
  --separation_multiplier=2. \
  --batch_size=64 \
  --eval_batch_size=256 \
  --lr=1e-3 \
  --checkpoint_every=2500 \
  --summarize_every=2500 \
  --num_steps=100000000 \
  --logdir=gs://l2i/k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=16 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=4 \
#  --max_k=4 \
#  --data_points_per_mode=25 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=16 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=16 \
#  --max_k=16 \
#  --data_points_per_mode=25 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/k_sweep


#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=32 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=32 \
#  --max_k=32 \
#  --data_points_per_mode=25 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=64 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=64 \
#  --max_k=64 \
#  --data_points_per_mode=25 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=128 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=128 \
#  --max_k=128 \
#  --data_points_per_mode=25 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/k_sweep
