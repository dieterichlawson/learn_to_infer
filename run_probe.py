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
import train

from absl import app
from absl import flags

import jax
from jax.config import config
import jax.numpy as jnp

import flax
from flax import optim
from flax.training import checkpoints

import scipy as oscipy
import matplotlib.pyplot as plt


flags.DEFINE_boolean("attach_probe", True,
                     "Load a model and initialize a new probde. If false will attempt to load "
                     "a combined model and probe from logdir.")
flags.DEFINE_integer("num_encoders", 6,
                     "Number of encoder modules in the transformer.")
flags.DEFINE_integer("num_decoders", 6,
                     "Number of decoder modules in the transformer.")
flags.DEFINE_integer("num_heads", 8,
                     "Number of attention heads in the transformer.")
flags.DEFINE_integer("key_dim", 32,
                     "The dimension of the keys in the transformer.")
flags.DEFINE_integer("value_dim_per_head", 32,
                     "The dimension of the values in the transformer "
                     "for each head.")
flags.DEFINE_integer("data_dim", 2,
                     "The dimension of the points to cluster.")
flags.DEFINE_integer("k", None,
                     "The number of modes in the data. If provided, overrides min_k and max_k.")
flags.DEFINE_integer("min_k", 2,
                     "The minimum number of modes in the data.")
flags.DEFINE_integer("max_k", 10,
                     "the maximum number of modes in the data.")
flags.DEFINE_float("noise_pct", None,
                   "The percentage of the datasets that will be noise points.")
flags.DEFINE_enum("dist", "l2", ["l2", "kl", "symm_kl"],
                  "The distance function used to measure similarity of components in the loss.")
flags.DEFINE_integer("data_points_per_mode", 25,
                     "Number of data points to include per mode in the data.")
flags.DEFINE_integer("cov_dof", None,
                     "Degrees of freedom in sampling the random covariances.")
flags.DEFINE_enum("cov_prior", "inv_wishart",
                  ["inv_wishart", "wishart"],
                  "The prior to use for the covariance matrix.")
flags.DEFINE_float("mode_var", 1.,
                   "The variance of the modes in the GMM used when "
                   "not sampling.")
flags.DEFINE_float("dist_multiplier", .95,
                   "Confidence interval that will be nonoverlapping when sampling the meaans")
flags.DEFINE_boolean("parallel", True,
                     "If possible, train in parallel across devices.")
flags.DEFINE_integer("batch_size", 64,
                     "The batch size.")
flags.DEFINE_integer("eval_batch_size", 256,
                     "The batch size for evaluation.")
flags.DEFINE_integer("num_steps", int(1e6),
                     "The number of steps to train for.")
flags.DEFINE_float("probe_lr", 1e-3,
                   "The learning rate for the probe.")
flags.DEFINE_float("lr", 1e-3,
                   "The learning rate for ADAM.")
flags.DEFINE_integer("summarize_every", 100,
                     "Number of steps between summaries.")
flags.DEFINE_integer("expensive_summarize_every", 10000,
                     "Number of steps between expensive summaries.")
flags.DEFINE_integer("checkpoint_every", 5000,
                     "Number of steps between checkpoints.")
flags.DEFINE_string("logdir", "/tmp/transformer",
                    "The directory to put summaries and checkpoints.")
flags.DEFINE_string("og_logdir", "/tmp/transformer",
                    "The directory containing the checkpoint to probe.")
flags.DEFINE_string("tag", "",
                    "String to append to the logdir.")

FLAGS = flags.FLAGS

def attach_probe(
    key,
    logdir,
    num_encoders=4,
    num_decoders=4,
    num_heads=8,
    value_dim=128,
    data_points_per_mode=25,
    max_k=10,
    data_dim=2):
  key1, key2 = jax.random.split(key)
  old_model = gmm_models.MSWUnconditional(
      data_dim=data_dim, max_k=max_k, algo_k=max_k, num_heads=num_heads,
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
      data_dim=data_dim, max_k=max_k, algo_k=max_k, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=value_dim,
      normalization="layer_norm", dist="l2")
  new_init_params = new_model.init_params(key2)

  old_params_flat = flax.traverse_util.flatten_dict(old_opt.target)
  new_params_flat = flax.traverse_util.flatten_dict(new_init_params)

  for k, v in old_params_flat.items():
    assert k in new_params_flat.keys()
    assert old_params_flat[k].shape == new_params_flat[k].shape
    new_params_flat[k] = v
  
  return new_model, flax.traverse_util.unflatten_dict(new_params_flat)

def make_probe(
    key,
    num_encoders=4,
    num_decoders=4,
    num_heads=8,
    value_dim=128,
    data_points_per_mode=25,
    max_k=10,
    data_dim=2):
 
  model = gmm_models.ProbedMSWUnconditional(
      data_dim=data_dim, max_k=max_k, algo_k=max_k, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=value_dim,
      normalization="layer_norm", dist="l2")
  init_params = new_model.init_params(key)

  return model, init_params


def make_loss(model,
              min_k=2,
              max_k=10,
              noise_pct=None,
              data_points_per_mode=25,
              cov_dof=10,
              cov_prior="inv_wishart",
              mode_var=1.,
              dist_mult=2.,
              data_dim=2,
              batch_size=128):

  def sample_train_batch(key):
    return sample_gmm.sample_batch_random_ks(
          key, "mean_scale_weight", batch_size, min_k, max_k, max_k*data_points_per_mode,
          data_dim, mode_var, cov_dof, cov_prior, dist_mult, noise_pct)

  def loss(params, key):
    key, subkey = jax.random.split(key)
    xs, _, ks, mog_params = sample_train_batch(key)
    losses, _, _ = model.loss(
        params, xs, ks*data_points_per_mode, mog_params, ks, subkey)
    return jnp.mean(losses)

  return loss

def make_summarize(
    model,
    min_k=2,
    max_k=10,
    noise_pct=None,
    data_points_per_mode=50,
    cov_dof=10,
    cov_prior="inv_wishart",
    dist_mult=2.,
    data_dim=2,
    mode_var=1.,
    eval_batch_size=256):
  
  def sample_eval_batch(key, points_per_mode, min_k, max_k):
    xs, cs, ks, params = sample_gmm.sample_batch_random_ks(
            key, "mean_scale_weight", eval_batch_size, min_k, max_k, 
            max_k * points_per_mode, data_dim, mode_var, cov_dof, cov_prior, 
            dist_mult, noise_pct)
    return xs, cs, ks, params

  def sample_and_pred(key, params, points_per_mode, min_k, max_k):
    xs, cs, ks, true_gmm_params = sample_eval_batch(key, points_per_mode, min_k, max_k)
    losses, entropy, guess_ce = model.loss(params, xs, ks*points_per_mode, true_gmm_params, ks, key)
    # [num_layers]
    return jnp.mean(losses, axis=-1), jnp.mean(entropy), jnp.mean(guess_ce)

  sample_and_pred = jax.jit(sample_and_pred, static_argnums=(2,3,4))

  def summarize(writer, step, params, key):
    losses, entropy, guess_ce = sample_and_pred(key, params, data_points_per_mode, min_k, max_k)
    print("Entropy: %0.2f" % entropy)
    print("Guessing CE: %0.4f" % guess_ce)
    for i in range(losses.shape[0]):
      writer.scalar("layer %d cross-entropy" % (i+1), losses[i], step=step)
      print("layer %d cross-entropy" % (i+1), losses[i])

  return summarize


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
        config.dist_multiplier, config.data_dim, config.min_k, config.max_k, 
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

  if FLAGS.k is not None:
    FLAGS.min_k = FLAGS.k
    FLAGS.max_k = FLAGS.k

  if FLAGS.parallel and train.can_train_parallel():
    assert FLAGS.batch_size % jax.local_device_count(
    ) == 0, "Device count must evenly divide batch_size"
    FLAGS.batch_size = int(FLAGS.batch_size / jax.local_device_count())

  FLAGS.dist_multiplier = oscipy.stats.chi2.ppf(FLAGS.dist_multiplier, df=FLAGS.data_dim)

  og_dirname = make_dirname(FLAGS)
  og_logdir = os.path.join(FLAGS.og_logdir, og_dirname)
  probe_dirname = "probe_" + og_dirname
  probe_logdir = os.path.join(FLAGS.logdir, probe_dirname)
  
  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  if FLAGS.attach_probe:
    model, init_params = attach_probe(
      key,
      og_logdir,
      num_encoders=FLAGS.num_encoders,
      num_decoders=FLAGS.num_decoders,
      num_heads=FLAGS.num_heads,
      value_dim=FLAGS.value_dim_per_head*FLAGS.num_heads,
      data_points_per_mode=FLAGS.data_points_per_mode,
      max_k=FLAGS.max_k,
      data_dim=FLAGS.data_dim)
  else:
    model, init_params = make_model(
      key,
      num_encoders=FLAGS.num_encoders,
      num_decoders=FLAGS.num_decoders,
      num_heads=FLAGS.num_heads,
      value_dim=FLAGS.value_dim_per_head*flags.num_heads,
      data_points_per_mode=FLAGS.data_points_per_mode,
      max_k=FLAGS.max_k,
      data_dim=FLAGS.data_dim)
  loss_fn = make_loss(
      model,
      min_k=FLAGS.min_k,
      max_k=FLAGS.max_k,
      noise_pct=FLAGS.noise_pct,
      data_points_per_mode=FLAGS.data_points_per_mode,
      cov_dof=FLAGS.cov_dof,
      cov_prior=FLAGS.cov_prior,
      mode_var=FLAGS.mode_var,
      dist_mult=FLAGS.dist_multiplier,
      data_dim=FLAGS.data_dim,
      batch_size=FLAGS.batch_size)
  summarize_fn = make_summarize(
    model,
    min_k=FLAGS.min_k,
    max_k=FLAGS.max_k,
    noise_pct=FLAGS.noise_pct,
    data_points_per_mode=FLAGS.data_points_per_mode,
    cov_dof=FLAGS.cov_dof,
    cov_prior=FLAGS.cov_prior,
    dist_mult=FLAGS.dist_multiplier,
    data_dim=FLAGS.data_dim,
    mode_var=FLAGS.mode_var,
    eval_batch_size=FLAGS.eval_batch_size)
  train.train_loop(
      subkey,
      init_params,
      loss_fn,
      parallel=FLAGS.parallel,
      lr=FLAGS.probe_lr,
      num_steps=FLAGS.num_steps,
      summarize_fn=summarize_fn,
      summarize_every=FLAGS.summarize_every,
      checkpoint_every=FLAGS.checkpoint_every,
      clobber_checkpoint=False,
      logdir=probe_logdir)

if __name__ == "__main__":
  app.run(main)
