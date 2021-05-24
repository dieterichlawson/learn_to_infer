# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/bin/bash
set -e
set -x

python3 run_tsne.py \
  --data_dim=4 \
  --num_encoders=12 \
  --num_decoders=2 \
  --num_heads=16 \
  --min_k=8 \
  --max_k=8 \
  --data_points_per_mode=155 \
  --cov_prior=inv_wishart \
  --cov_dof=6 \
  --dist_multiplier=0.68 \
  --dist=l2 \
  --batch_size=16 \
  --lr=0.1 \
  --og_logdir=gs://l2i/deep_exp_2 \
  --tag=9
