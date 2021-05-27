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
import pickle


flags.DEFINE_string("exp", None, "Experiment to evaluate.")
flags.DEFINE_integer("eval_batch_size", 256,
                     "The batch size for evaluation.")
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
      #"mean_scale": gmm_models.MeanScaleInferenceMachine,
      "mean_scale_weight": gmm_models.MSWOriginal,
      "msw_unconditional": gmm_models.MSWUnconditional,
      "mean_unconditional": gmm_models.UnconditionalMeanInferenceMachine,
      "fixed_k": gmm_models.UnconditionalFixedK,
}

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
          2 * max_k * points_per_mode, data_dim, mode_var, cov_dof, cov_prior, dist_mult, None)
    else:
      ks = jnp.full([eval_batch_size], max_k)
      xs, cs, params = sample_gmm.sample_batch_fixed_ks(
        key, sampling_types[model_name], ks, max_k, 2 * max_k * points_per_mode, 
        data_dim, mode_var, cov_dof, cov_prior, dist_mult, None)
    train_xs = xs[:, :max_k * points_per_mode]
    test_xs = xs[:, max_k * points_per_mode:]
    train_cs = cs[:, :max_k * points_per_mode]
    test_cs = cs[:, max_k * points_per_mode:]
    return train_xs, test_xs, train_cs, test_cs, ks, params

  def model_classify(params, inputs, ks, points_per_mode):
    return gmm_models.classify_with_defaults(
        model, params, inputs, eval_batch_size, ks*points_per_mode, ks,
        max_k, jnp.eye(data_dim)*mode_var)

  def sample_and_classify_eval_batch(key, params, points_per_mode, min_k, max_k):
    train_xs, test_xs, train_cs, test_cs, ks, true_gmm_params = sample_eval_batch(key,
        points_per_mode, min_k, max_k)
    tfmr_train_cs, tfmr_gmm_params = model_classify(params, train_xs, ks, points_per_mode)
    tfmr_test_cs = jax.vmap(gmm_models.masked_classify_points)(
            test_xs, tfmr_gmm_params[0], tfmr_gmm_params[1], tfmr_gmm_params[2], ks)
    return (train_xs, test_xs, tfmr_train_cs, train_cs, tfmr_test_cs, test_cs, ks, 
            true_gmm_params, tfmr_gmm_params)
  
  sample_and_classify_eval_batch = jax.jit(sample_and_classify_eval_batch, static_argnums=(2,3,4))

  train_xs, test_xs, train_cs, test_cs, ks, _ = sample_eval_batch(
      key, data_points_per_mode, min_k, max_k)

  em_metrics = gmm_eval.compute_baseline_metrics(
      key, train_xs, train_cs, test_xs, test_cs, min_k)
  
  tfmr_train_cs, tfmr_gmm_params = model_classify(params, train_xs, ks, data_points_per_mode)
  tfmr_test_cs = jax.vmap(gmm_models.masked_classify_points)(
            test_xs, tfmr_gmm_params[0], tfmr_gmm_params[1], tfmr_gmm_params[2], ks)

  tfmr_train_accs, tfmr_train_f1s, tfmr_train_lls = gmm_eval.batch_metrics(
        train_xs, tfmr_gmm_params, train_cs, tfmr_train_cs, ks*data_points_per_mode, ks)
  tfmr_test_accs, tfmr_test_f1s, tfmr_test_lls = gmm_eval.batch_metrics(
        test_xs, tfmr_gmm_params, test_cs, tfmr_test_cs, ks*data_points_per_mode, ks)

  tfmr_metrics = (tfmr_train_accs, tfmr_test_accs, tfmr_train_f1s, tfmr_test_f1s, 
      tfmr_train_lls, tfmr_test_lls)
  return tfmr_metrics, em_metrics


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
  minibatch_size = eval_batch_size // num_batches

  tfmr_metrics = onp.zeros([eval_batch_size, 6])
  em_metrics = onp.zeros([eval_batch_size, 6])

  eval_fn = lambda k: eval_model(k, model, params, model_name, min_k, max_k, data_points_per_mode,
      cov_dof, cov_prior, dist_mult, data_dim, mode_var, minibatch_size)
  eval_fn = jax.jit(eval_fn)

  for i in range(num_batches):
    key, subkey = jax.random.split(key)
    tfmr_ms, em_ms = eval_fn(subkey)

    #tfmr_ms, em_ms = eval_model(k1, model, params, model_name, min_k, max_k, 
    #    data_points_per_mode, cov_dof, cov_prior, dist_mult, data_dim, mode_var, 
    #    minibatch_size)
    tfmr_metrics[i*minibatch_size:(i+1)*minibatch_size] = onp.array(tfmr_ms).T
    em_metrics[i*minibatch_size:(i+1)*minibatch_size] = onp.array(em_ms).T

  return tfmr_metrics, em_metrics
  

def print_tables(vals, num_digits=2):

  def make_table(em_or_tfmr, index, include_ddim=True):
    table = []
    for i, ddim in enumerate([2, 4, 8, 16]):
      row = []
      if include_ddim:
        row.append(ddim)
      for j,k in enumerate([2 ,4, 8, 16]):
        if em_or_tfmr == "tfmr":
          row.append("%0.3f" % vals[ddim][k][0][index])
        elif em_or_tfmr == "em":
          row.append("%0.3f" % vals[ddim][k][1][index])
        else:
          assert False, "Bad em or tfmr"
      table.append(row)
    if include_ddim:
      return tabulate(table, headers=["data dim", "k=2", "k=4", "k=8", "k=16"])
    else:
      return tabulate(table, headers=["k=2", "k=4", "k=8", "k=16"])

  for i, name in enumerate(["Train Acc", "Test Acc", "Train F1", "Test F1", "Train LL", "Test LL"]):
    print(name)
    print("Transformer                              EM")
    tfmr_table = make_table("tfmr", i)
    em_table = make_table("em", i, include_ddim=False)
    for a, b in zip(tfmr_table.split('\n'), em_table.split('\n')):
      print(a, "|", b)


def make_logdir(config):
  basedir = config.logdir
  exp_dir = (
      "%s"
      "_nheads_%d"
      "_nenc_%d"
      "_ndec_%d"
      "_sepm_%0.2f"
      "_data_dim_%d"
      "_mink_%d"
      "_maxk_%d"
      "_dps_per_k_%d"
      "_cov_prior_%s"
      "_cov_dof_%d"
      "_%s"
      "_dist_%s"
      "_lr_%0.3f"
      "_tpu%s" % (
        config.model_name,
        config.num_heads, config.num_encoders, config.num_decoders, 
        config.dist_multiplier, config.data_dim, config.min_k, config.max_k,
        config.data_points_per_mode, config.cov_prior, 
        config.cov_dof, config.normalization, config.dist, config.lr, config.tag)
      )
  return os.path.join(basedir, exp_dir)

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
  metrics = defaultdict(dict)
  for config in configs:
    model, params = load_model(config)
    key, k1 = jax.random.split(key)
    metrics[config.data_dim][config.min_k] = eval_model_in_batches(k1, model, params,
        config.model_name,
        config.min_k,
        config.max_k,
        config.data_points_per_mode,
        config.cov_dof,
        config.cov_prior,
        config.dist_multiplier,
        config.data_dim,
        config.mode_var,
        FLAGS.eval_batch_size,
        FLAGS.num_batches)
  f = open("out.pkl", 'wb')
  pickle.dump(metrics, f)
  f.close()

  #print_tables(metrics, metrics)


if __name__ == "__main__":
  app.run(main)
