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

python3 run_probe.py \
  --data_dim=2 \
  --num_encoders=4 \
  --num_decoders=2 \
  --num_heads=2 \
  --min_k=2 \
  --max_k=2 \
  --data_points_per_mode=50 \
  --cov_prior=inv_wishart \
  --cov_dof=4 \
  --dist_multiplier=0.68 \
  --dist=l2 \
  --batch_size=16 \
  --eval_batch_size=32 \
  --num_steps=100000 \
  --summarize_every=500 \
  --expensive_summarize_every=500 \
  --checkpoint_every=500 \
  --lr=2.0
