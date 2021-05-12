# coding=utf-8
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

"""Runner for transformer GMM experiments.
"""
import os
from types import SimpleNamespace
from collections import defaultdict
import importlib

from tabulate import tabulate

from bin import orchestration as orch
from run_gmm import make_logdir
import sample_gmm
import gmm_eval
import gmm_models
import plotting
import util

from absl import app
from absl import flags

import jax
from jax.config import config
import jax.numpy as jnp
import numpy as onp

import scipy as oscipy
import matplotlib.pyplot as plt


flags.DEFINE_string("exp", None, "Experiment to evaluate.")
flags.DEFINE_string("eval_points_per_mode", None, "List of numbers of points per mode to eval.")
#flags.DEFINE_integer("eval_batch_size", 256,
#                     "The batch size for evaluation.")
flags.DEFINE_integer("num_batches", 8,
                     "The number of batches to split eval_batch_size up into.")

FLAGS = flags.FLAGS

sampling_types = {
 "mean": "mean",
 "mean_unconditional": "mean",
 "mean_scale": "mean_scale",
 "mean_scale_weight": "mean_scale_weight",
 "msw_unconditional": "mean_scale_weight",
 "fixed_k": "mean_scale_weight"
}

class_dict = {
      "mean": gmm_models.OriginalMeanInferenceMachine,
      "mean_scale_weight": gmm_models.MSWOriginal,
      "msw_unconditional": gmm_models.MSWUnconditional,
      "mean_unconditional": gmm_models.UnconditionalMeanInferenceMachine,
      "fixed_k": gmm_models.UnconditionalFixedK,
}

METRICS = ["Train Acc", "Test Acc", "Train F1", "Test F1", "Train LL", "Test LL"]

def eval_model(
  key,
  model,
  params,
  model_name,
  min_k,
  max_k,
  data_points_per_mode,
  cov_dof,
  cov_prior,
  dist_mult,
  data_dim,
  mode_var,
  eval_batch_size):

  def sample_eval_batch(key, points_per_mode, min_k, max_k):
    if min_k != max_k:
      xs, cs, ks, params = sample_gmm.sample_batch_random_ks(
          key, sampling_types[model_name], eval_batch_size, min_k, max_k, 
          2 * max_k * points_per_mode, data_dim, mode_var, cov_dof, cov_prior, dist_mult)
    else:
      ks = jnp.full([eval_batch_size], max_k)
      xs, cs, params = sample_gmm.sample_batch_fixed_ks(
        key, sampling_types[model_name], ks, max_k, 2 * max_k * points_per_mode, 
        data_dim, mode_var, cov_dof, cov_prior, dist_mult)
    train_xs = xs[:, :max_k * points_per_mode]
    test_xs = xs[:, max_k * points_per_mode:]
    train_cs = cs[:, :max_k * points_per_mode]
    test_cs = cs[:, max_k * points_per_mode:]
    return train_xs, test_xs, train_cs, test_cs, ks, params

  def model_classify(params, inputs, ks, points_per_mode):
    return gmm_models.classify_with_defaults(
        model, params, inputs, eval_batch_size, ks*points_per_mode, ks,
        max_k, jnp.eye(data_dim)*mode_var)

  train_xs, test_xs, train_cs, test_cs, ks, _ = sample_eval_batch(
      key, data_points_per_mode, min_k, max_k)

  #em_metrics = gmm_eval.compute_masked_baseline_metrics(
  #    train_xs, train_cs, test_xs, test_cs, sampling_types[model_name], mode_var, 
  #    ks, ks*data_points_per_mode)
  
  tfmr_train_cs, tfmr_gmm_params = model_classify(params, train_xs, ks, data_points_per_mode)
  tfmr_test_cs = jax.vmap(gmm_models.masked_classify_points)(
            test_xs, tfmr_gmm_params[0], tfmr_gmm_params[1], tfmr_gmm_params[2], ks)

  tfmr_train_acc, tfmr_train_f1, tfmr_train_ll = gmm_eval.compute_metrics(
        train_xs, tfmr_gmm_params, train_cs, tfmr_train_cs, ks*data_points_per_mode, ks)
  tfmr_test_acc, tfmr_test_f1, tfmr_test_ll = gmm_eval.compute_metrics(
        test_xs, tfmr_gmm_params, test_cs, tfmr_test_cs, ks*data_points_per_mode, ks)
  tfmr_metrics = (tfmr_train_acc, tfmr_test_acc, tfmr_train_f1, tfmr_test_f1, 
      tfmr_train_ll, tfmr_test_ll)
  return tfmr_metrics#, em_metrics

def eval_model_in_batches(
  key,
  model,
  params,
  model_name,
  min_k,
  max_k,
  data_points_per_mode,
  cov_dof,
  cov_prior,
  dist_mult,
  data_dim,
  mode_var,
  eval_batch_size,
  num_batches):
 
  assert eval_batch_size % num_batches == 0

  tfmr_metrics = onp.zeros([6])

  for i in range(num_batches):
    key, k1 = jax.random.split(key)

    tfmr_metrics += eval_model(k1, model, params, model_name, min_k, max_k, 
        data_points_per_mode, cov_dof, cov_prior, dist_mult, data_dim, mode_var, 
        eval_batch_size // num_batches)

  return tfmr_metrics / num_batches

def print_tables(metrics, eval_data_points=[12, 25, 50, 100, 200]):
  for data_dim in [2, 4, 8]:
    print("Data dim %d" % data_dim)
    for i, metric_name in enumerate(METRICS):
      print(metric_name)
      table = []
      for train_dppm in [12, 25, 50, 100, 200]:
        row = [train_dppm]
        for eval_dppm in eval_data_points:
          row.append(metrics[data_dim][train_dppm][eval_dppm][i])
        table.append(row)
      print(tabulate(table, 
        headers=["Train DPPM", "Eval DPPM=12", "25", "50", "100", "200"]))

def normalize_configs(cs):
  for i, c in enumerate(cs):
    if "min_k" not in dir(c):
      c.min_k = c.k
    if "max_k" not in dir(c):
      c.max_k = c.k
    c.tag = str(i+1)
    c.dist_multiplier = oscipy.stats.chi2.ppf(c.dist_multiplier, df=c.data_dim)
    if "mode_var" not in dir(c):
      c.mode_var = 1.
    if "cov_dof" not in dir(c):
      c.cov_dof = c.data_dim + 2

def load_model(c):
  model = class_dict[c.model_name](
      data_dim=c.data_dim, max_k=c.max_k, algo_k=c.max_k, num_heads=c.num_heads,
      num_encoders=c.num_encoders, num_decoders=c.num_decoders,
      qkv_dim=c.value_dim_per_head*c.num_heads,
      normalization=c.normalization, dist='l2')
  # We use a throwaway key because we won't use these parameters.
  params = model.init_params(jax.random.PRNGKey(0))
  params = util.load_parameters(make_logdir(c), params)
  return model, params

def main(unused_argv):
  key = jax.random.PRNGKey(0)
  hparams = orch.load_exp_hparams(importlib.import_module("bin.%s" % FLAGS.exp))
  configs = [SimpleNamespace(**d) for d in hparams]
  normalize_configs(configs)
  metrics = defaultdict(lambda: defaultdict(dict))
 
  if FLAGS.eval_points_per_mode is not None:
    FLAGS.eval_points_per_mode = util.parse_int_list(FLAGS.eval_points_per_mode)
  else:
    FLAGS.eval_points_per_mode = list(set([c.data_points_per_mode for c in configs]))
  
  for config in configs:
    model, params = load_model(config)
    metrics[config.data_dim][config.data_points_per_mode] = {}
    for dppm in FLAGS.eval_points_per_mode:
      key, k1 = jax.random.split(key)
      metrics[config.data_dim][config.data_points_per_mode][dppm] = eval_model_in_batches(
          k1, model, params,
          config.model_name,
          config.min_k,
          config.max_k,
          dppm,
          config.cov_dof,
          config.cov_prior,
          config.dist_multiplier,
          config.data_dim,
          config.mode_var,
          FLAGS.eval_batch_size,
          FLAGS.num_batches)
  print_tables(metrics)

if __name__ == "__main__":
  app.run(main)
