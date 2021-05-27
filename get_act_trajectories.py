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

import sample_gmm
import gmm_eval
import gmm_models
import plotting
import util
import em

from absl import app
from absl import flags

import jax
from jax import vmap
import jax.numpy as jnp

import flax
from flax import optim
from flax.training import checkpoints

import numpy as onp
import scipy as oscipy

flags.DEFINE_integer(
    "num_encoders", 6,
    "Number of encoder modules in the transformer.")
flags.DEFINE_integer(
    "num_decoders", 6,
    "Number of decoder modules in the transformer.")
flags.DEFINE_integer(
    "num_heads", 8,
    "Number of attention heads in the transformer.")
flags.DEFINE_integer(
    "key_dim", 32,
    "The dimension of the keys in the transformer.")
flags.DEFINE_integer(
    "value_dim_per_head", 32,
    "The dimension of the values in the transformer for each head.")
flags.DEFINE_integer(
    "data_dim", 2,
    "The dimension of the points to cluster.")
flags.DEFINE_integer(
    "k", None,
    "The number of modes in the data.")
flags.DEFINE_float(
    "noise_pct", None,
    "The percentage of the datasets that will be noise points.")
flags.DEFINE_enum(
    "dist", "l2", ["l2", "kl", "symm_kl"],
    "The distance function used to measure similarity of components in the loss.")
flags.DEFINE_integer(
    "data_points_per_mode", 25,
    "Number of data points to include per mode in the data.")
flags.DEFINE_integer(
    "cov_dof", None,
    "Degrees of freedom in sampling the random covariances.")
flags.DEFINE_enum(
    "cov_prior", "inv_wishart", ["inv_wishart", "wishart"],
    "The prior to use for the covariance matrix.")
flags.DEFINE_float(
    "mode_var", 1.,
    "The variance of the modes in the GMM used when not sampling.")
flags.DEFINE_float(
    "dist_multiplier", .95,
    "Confidence interval that will be nonoverlapping when sampling the meaans")
flags.DEFINE_integer(
    "batch_size", 64,
    "The batch size.")
flags.DEFINE_float(
    "lr", 1e-3,
    "The learning rate for ADAM.")
flags.DEFINE_string(
    "logdir", "/tmp/transformer",
    "The directory to put summaries and checkpoints.")
flags.DEFINE_string(
    "og_logdir", "/tmp/transformer",
    "The directory containing the checkpoint to probe.")
flags.DEFINE_string(
    "tag", "",
    "String to append to the logdir.")
flags.DEFINE_float(
    "em_tol", 1e-5,
    "Tolerance for EM.")
flags.DEFINE_float(
    "em_reg", 1e-4,
    "Regularization for EM.")
flags.DEFINE_integer(
    "max_em_steps", 200,
    "Max number of steps EM can take.")

FLAGS = flags.FLAGS

def attach_probe(
    key,
    logdir,
    num_encoders=4,
    num_decoders=4,
    num_heads=8,
    value_dim=128,
    data_points_per_mode=25,
    k=10,
    data_dim=2,
    batch_size=2):
  key1, key2 = jax.random.split(key)
  old_model = gmm_models.MSWUnconditional(
      data_dim=data_dim, max_k=k, algo_k=k, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=value_dim,
      normalization="layer_norm", dist="l2")
  init_params = old_model.init_params(key1)
  opt_def = optim.Adam()
  old_opt = opt_def.create(init_params)

  if util.has_checkpoint(logdir):
    print("Loading original model checkpoint from %s" % logdir)
    old_opt = checkpoints.restore_checkpoint(logdir, old_opt)
  else:
    assert False, "No checkpoint found in %s" % logdir
 
  new_model = gmm_models.ProbedMSWUnconditional(
      batch_size=batch_size,
      data_dim=data_dim, max_k=k, algo_k=k, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=value_dim,
      normalization="layer_norm", dist="l2")
  new_init_params = new_model.init_params(key2)

  old_params_flat = flax.traverse_util.flatten_dict(old_opt.target)
  new_params_flat = flax.traverse_util.flatten_dict(new_init_params)

  for k, v in old_params_flat.items():
    assert k in new_params_flat.keys(), "Key %s not in %s" % (k, new_params_flat.keys())
    assert old_params_flat[k].shape == new_params_flat[k].shape
    new_params_flat[k] = v
  
  return new_model, flax.traverse_util.unflatten_dict(new_params_flat)

def run_probe(
    key,
    model,
    params,
    k=2,
    noise_pct=None,
    data_points_per_mode=50,
    cov_dof=10,
    cov_prior="inv_wishart",
    dist_mult=2.,
    data_dim=2,
    mode_var=1.,
    batch_size=256,
    em_tol=1e-5,
    em_reg=1e-4,
    max_em_steps=200):
  
  xs, cs, ks, true_gmm_params = sample_gmm.sample_batch_random_ks(
          key, "mean_scale_weight", batch_size, k, k, 
          k * data_points_per_mode, data_dim, mode_var, cov_dof, cov_prior, 
          dist_mult, noise_pct)

  _, _, _, _, _, activations, queries, keys, attn_weights, log_resps = model.loss(
      params, xs, ks*data_points_per_mode, true_gmm_params, ks, key)

  batch_em = vmap(em.em, in_axes=(0, None, None, 0, None, None))
  em_params, em_num_steps, em_resps, final_elbo,_ = batch_em(
      xs, k, max_em_steps, jax.random.split(key, num=batch_size), em_tol, em_reg)
  em_resps = em_resps[:,:jnp.max(em_num_steps)]
  return xs, cs, log_resps, activations, queries, keys, attn_weights, em_resps, em_num_steps


def make_dirname(config):
  exp_dir = (
      "%s"
      "_nheads_%d"
      "_nenc_%d"
      "_ndec_%d"
      "_sepm_%0.2f"
      "_data_dim_%d"
      "_mink_%d"
      "_maxk_%d"
      "_nsy_%0.2f"
      "_dps_per_k_%d"
      "_cov_prior_%s"
      "_cov_dof_%d"
      "_%s"
      "_dist_%s"
      "_lr_%0.3f"
      "_tpu%s" % (
       "msw_unconditional", 
        config.num_heads, config.num_encoders, config.num_decoders, 
        config.dist_multiplier, config.data_dim, config.k, config.k, 
        config.noise_pct or 0.,
        config.data_points_per_mode, config.cov_prior, 
        config.cov_dof, "layer_norm", "l2", config.lr, config.tag)
      )
  return exp_dir

def load_model_params(model_logdir, model, init_params):
  optimizer_def = optim.Adam()
  optimizer = optimizer_def.create(init_params)
  optimizer = load_probe_checkpoint(logdir, optimizer)
  return optimizer

def main(unused_argv):
  if FLAGS.cov_dof is None:
    FLAGS.cov_dof = FLAGS.data_dim + 2

  assert FLAGS.cov_dof >= FLAGS.data_dim + 2, "Wishart DOF must be >= 2 + data dim."


  FLAGS.dist_multiplier = oscipy.stats.chi2.ppf(FLAGS.dist_multiplier, df=FLAGS.data_dim)

  og_dirname = make_dirname(FLAGS)
  og_logdir = os.path.join(FLAGS.og_logdir, og_dirname)
  probe_dirname = "probe_" + og_dirname
  probe_logdir = os.path.join(FLAGS.logdir, probe_dirname)
  
  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model, params = attach_probe(
      key,
      og_logdir,
      num_encoders=FLAGS.num_encoders,
      num_decoders=FLAGS.num_decoders,
      num_heads=FLAGS.num_heads,
      value_dim=FLAGS.value_dim_per_head*FLAGS.num_heads,
      data_points_per_mode=FLAGS.data_points_per_mode,
      k=FLAGS.k,
      data_dim=FLAGS.data_dim,
      batch_size=FLAGS.batch_size)
  xs, cs, log_resps, activations, queries, keys, attn_weights, em_resps, em_num_steps = run_probe(
    subkey,
    model,
    params,
    k=FLAGS.k,
    noise_pct=FLAGS.noise_pct,
    data_points_per_mode=FLAGS.data_points_per_mode,
    cov_dof=FLAGS.cov_dof,
    cov_prior=FLAGS.cov_prior,
    dist_mult=FLAGS.dist_multiplier,
    data_dim=FLAGS.data_dim,
    mode_var=FLAGS.mode_var,
    batch_size=FLAGS.batch_size,
    em_tol=FLAGS.em_tol,
    em_reg=FLAGS.em_reg,
    max_em_steps=FLAGS.max_em_steps)

  onp.savez('probe.npz', 
      xs=xs, log_resps=log_resps, cs=cs, tfmr_acts=activations, tfmr_queries=queries, tfmr_keys=keys, 
      tfmr_attn_weights=attn_weights, true_cs=cs, 
      em_resps=em_resps, em_num_steps=em_num_steps)


if __name__ == "__main__":
  app.run(main)
