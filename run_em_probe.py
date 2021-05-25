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

import em
import sample_gmm
import train

from absl import app
from absl import flags

import jax
from jax import vmap
from jax.config import config
import jax.numpy as jnp
import jax.scipy as jscipy

import flax

import tensorflow_probability
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

import scipy as oscipy

flags.DEFINE_integer(
    "data_dim", 2, 
    "The dimension of the points to cluster.")
flags.DEFINE_integer(
    "k", 2, 
    "The number of modes in the data.")
flags.DEFINE_integer(
    "num_data_points", 200, 
    "Number of data points to include in the datasets.")
flags.DEFINE_integer(
    "cov_dof", None, 
    "Degrees of freedom in sampling the random covariances.")
flags.DEFINE_float(
    "mode_var", 1., 
    "The variance of the modes in the GMM used when not sampling.")
flags.DEFINE_float(
    "dist_multiplier", .95, 
    "Confidence interval that will be nonoverlapping when sampling the meaans")
flags.DEFINE_float(
    "noise_pct", 0.0, 
    "Percent of the dataset that will be noise.")
flags.DEFINE_integer(
    "em_max_num_steps", 200,
    "Maximum number of steps EM is allowed to take")
flags.DEFINE_float(
    "em_tol", 1e-5,
    "Tolerance used to determine convergence in EM")
flags.DEFINE_float(
    "em_regularization", 1e-5,
    "Weight of identity matrix added to covariaances in EM.")
flags.DEFINE_integer(
    "num_summary_points", 10,
    "The number of em runs to summarize.")
flags.DEFINE_boolean(
    "parallel", True, 
    "If possible, train in parallel across devices.")
flags.DEFINE_integer(
    "batch_size", 64, 
    "The batch size.")
flags.DEFINE_integer(
    "num_steps", int(1e6), 
    "The number of steps to train for.")
flags.DEFINE_float(
    "lr", 1e-3,
    "The learning rate for the probe.")
flags.DEFINE_integer(
    "summarize_every", 1000,
    "Number of steps between summaries.")
flags.DEFINE_integer(
    "checkpoint_every", 5000,
    "Number of steps between checkpoints.")
flags.DEFINE_string(
    "logdir", "/tmp/transformer",
    "The directory to put summaries and checkpoints.")
flags.DEFINE_string(
    "tag", "",
    "String to append to the logdir.")

FLAGS = flags.FLAGS


def generate_data(
    key,
    batch_size,
    num_data_points,
    data_dim, 
    k, 
    mode_var,
    cov_dof,
    dist_mult,
    noise_pct,
    em_tol, 
    em_max_num_steps, 
    em_regularization):
  
  def compute_true_resps(X, params):
    mus, scales, log_ws = params
    covs = jnp.einsum("...ik,...jk->...ij", scales, scales)
    # The probability of each data point under each component, 
    # log p(x=x_i|z_i=k, theta) 
    # shape [n, k]
    log_p_xs = vmap(
        jscipy.stats.multivariate_normal.logpdf,
        in_axes=(None, 0, 0))(X, mus, covs)
    log_p_xs = log_p_xs.T
    
    # Joint probability of z and x given the data
    # log p(x=x_i, z_i=k|theta) = log p(x=x_i|z_i=k, theta) + log p(z_i=k|theta)
    # shape [n,k]
    log_joints = log_p_xs + log_ws[jnp.newaxis,...]
    
    # resps are log joints minus log marginal, i.e.
    # log p(z_i=k|x=x_i,theta) = log p(x=x_i, z_i=k|theta) - log p(x=x_i|theta)
    # log p(x=x_i|theta) = log ( sum_k p(x=x_i, z_i=k|theta) )
    # shape [n,k]
    resps = log_joints - jscipy.special.logsumexp(log_joints, axis=1, keepdims=True)
    return resps

  Xs, _, _, params = sample_gmm.sample_batch_random_ks(
      jax.random.PRNGKey(0), "mean_scale_weight", batch_size, k, k, num_data_points,
      data_dim, mode_var, cov_dof, "inv_wishart", dist_mult, noise_pct)

  # get true resps
  true_resps = vmap(compute_true_resps)(Xs, params)
  # get em resps
  batch_em = vmap(em.em, in_axes=(0, None, None, 0, None, None))
  em_params, em_num_steps, all_resps, em_elbo = batch_em(
      Xs, k, em_max_num_steps, jax.random.split(key, num=batch_size), em_tol, em_regularization)
  
  return Xs, params, true_resps, all_resps, em_num_steps

def make_model(
    key, 
    batch_size, 
    num_data_points, 
    data_dim, 
    k, 
    mode_var,
    cov_dof,
    dist_mult,
    noise_pct,
    em_max_num_steps, 
    em_tol, 
    em_regularization, 
    num_summary_points):

  key1, key2 = jax.random.split(key)

  # true_resps are [batch_size, num_data_points, k]
  # em_resps are [batch_size, max_num_steps, num_data_points, k]
  # em_num_steps is [batch_size]
  # NOTE: params (2nd arg) has scales in it, not covs!
  _, _, true_resps, em_resps, em_num_steps = generate_data(
    key1, batch_size, num_data_points, data_dim, k,
    mode_var, cov_dof, dist_mult, noise_pct,
    em_tol, em_max_num_steps, em_regularization)
  
  true_resp_dist = tfd.Categorical(logits=true_resps)

  max_em_steps = jnp.max(em_num_steps)
  # slice em_resps to [batch_size, max_em_steps, num_data_points, k]
  em_resps = em_resps[:,:max_em_steps,...]
  # make a mask that is [max_em_steps, batch_size]
  mask = jnp.arange(max_em_steps)[:, jnp.newaxis] <= em_num_steps[jnp.newaxis,:]

  # make the model
  # need a dense general that will go from [batch_size, max_em_steps, num_data_points, k]
  # to [batch_size, max_em_steps, num_data_points, k]
  # with a different kxk linear function for each batch element and step
  # so kernel of shape [batch_size, max_em_steps, k, k]
  # bias of shape [batch_size, max_em_steps, k]
  model = flax.nn.DenseGeneral.partial(features=k, batch_dims=(0,1))
  _, init_params = model.init(
      key2, jnp.zeros([batch_size, max_em_steps, num_data_points, k]))

  def loss(params):
    pred_resps = model.call(params, em_resps)
    # compute loss between pred_resps and true_resps
    # pred_resps is [batch_size, max_em_steps, num_data_points, k]
    # transpose pred_resps to make the kl calculations work out
    # pred_resps is now [max_em_steps, batch_size, num_data_points, k]
    pred_resps = jnp.transpose(pred_resps, [1,0,2,3])
    pred_resp_dist = tfd.Categorical(logits=pred_resps)
    # [max_em_steps, batch_size, num_data_points]
    kls = true_resp_dist.kl_divergence(pred_resp_dist)
    losses = kls*mask[:,:,jnp.newaxis]
    return jnp.mean(losses), losses

  def loss_fn(params, unused_key):
    return loss(params)[0]

  def summarize(sw, step, params, key):
    _, losses = loss(params)
    print("Step %d" % step)
    for i in range(num_summary_points):
      s = "  point %d:" % i
      for j in range(em_num_steps[i]):
        s += " %0.6f" % jnp.mean(losses[j,i])
      print(s)

  return init_params, loss_fn, summarize

def main(unused_argv):
  if FLAGS.cov_dof is None:
    FLAGS.cov_dof = FLAGS.data_dim + 2

  assert FLAGS.cov_dof >= FLAGS.data_dim + 2, "Wishart DOF must be >= 2 + data dim."

  if FLAGS.parallel and train.can_train_parallel():
    assert FLAGS.batch_size % jax.local_device_count(
    ) == 0, "Device count must evenly divide batch_size"
    FLAGS.batch_size = int(FLAGS.batch_size / jax.local_device_count())

  FLAGS.dist_multiplier = oscipy.stats.chi2.ppf(FLAGS.dist_multiplier, df=FLAGS.data_dim)

  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  init_params, loss_fn, summarize_fn = make_model(
      key,
      FLAGS.batch_size,
      FLAGS.num_data_points,
      FLAGS.data_dim,
      FLAGS.k, 
      FLAGS.mode_var,
      FLAGS.cov_dof,
      FLAGS.dist_multiplier,
      FLAGS.noise_pct,
      FLAGS.em_max_num_steps, 
      FLAGS.em_tol, 
      FLAGS.em_regularization, 
      FLAGS.num_summary_points)
  train.train_loop(
      subkey,
      init_params,
      loss_fn,
      lambda t: FLAGS.lr,
      parallel=FLAGS.parallel,
      num_steps=FLAGS.num_steps,
      summarize_fn=summarize_fn,
      summarize_every=FLAGS.summarize_every,
      checkpoint_every=FLAGS.checkpoint_every,
      clobber_checkpoint=False,
      logdir=FLAGS.logdir)

if __name__ == "__main__":
  app.run(main)
