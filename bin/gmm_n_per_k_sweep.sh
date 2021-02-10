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
  --data_points_per_mode=4 \
  --separation_multiplier=2. \
  --batch_size=64 \
  --eval_batch_size=256 \
  --lr=1e-3 \
  --checkpoint_every=2500 \
  --summarize_every=2500 \
  --num_steps=100000000 \
  --logdir=gs://l2i/n_per_k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=16 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=8 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/n_per_k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=16 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=16 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/n_per_k_sweep


#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=32 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=32 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/n_per_k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=64 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=64 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/n_per_k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=128 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=128 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/n_per_k_sweep

#python3 run_gmm.py \
#  --model_name="mean_scale_weight" \
#  --data_dim=2 \
#  --cov_dof=4 \
#  --num_encoders=2 \
#  --num_decoders=2 \
#  --num_heads=128 \
#  --key_dim=32 \
#  --value_dim_per_head=64 \
#  --min_k=3 \
#  --max_k=3 \
#  --data_points_per_mode=256 \
#  --separation_multiplier=2. \
#  --batch_size=64 \
#  --eval_batch_size=256 \
#  --lr=1e-3 \
#  --checkpoint_every=2500 \
#  --summarize_every=2500 \
#  --num_steps=100000000 \
#  --logdir=gs://l2i/n_per_k_sweep
