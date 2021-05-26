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

"""Functions for running EM on GMMs."""
from functools import partial

import util

import jax
from jax import jit
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jscipy
import jax.experimental.host_callback as hcb

import tensorflow_probability
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions


def em(X, k, T, key, tol, regularization):
  n, data_dim = X.shape
  # Initialize centroids using k-means++ scheme
  init_centroids = kmeans_pp_init(X, k, key)
  # [k, d, d]
  init_covs = jnp.array([jnp.cov(X, rowvar=False)]*k)
  # centroid mixture weights, [k]
  init_log_weights = -jnp.ones(k)*jnp.log(n)

  init_all_resps = jnp.zeros([T, n, k])

  def em_step(state):
    t, centroids, covs, log_weights, _, elbo, _, all_resps  = state
    # E step

    # The probability of each data point under each component, 
    # log p(x=x_i|z_i=k, theta) 
    # shape [n, k]
    log_p_xs = vmap(
        jscipy.stats.multivariate_normal.logpdf,
        in_axes=(None, 0, 0))(X, centroids, covs)
    log_p_xs = log_p_xs.T


    expanded_log_ws = log_weights[jnp.newaxis, :]

    # The marginal probability of each data point given the parameters
    # log(sum_k p(x=x_i|z_i=k, theta) p(z_i=k|theta)) = log p(x=x_i|theta)
    # shape [n, 1]
    log_marginal_xs = jscipy.special.logsumexp(
        log_p_xs + expanded_log_ws, axis=1, keepdims=True)

 
    # [n, k]
    # the responsibilities,
    # log p(x=x_i|z_i=k, theta) + log p(z_i=k|theta) - log p(x=x_i|theta)
    # = log p(z_i = k|x=x_i, theta)
    resps = log_p_xs + expanded_log_ws - log_marginal_xs

    model_dist = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(logits=log_weights),
        components_distribution=tfd.MultivariateNormalFullCovariance(
            loc=centroids, covariance_matrix=covs))
    new_elbo = jnp.mean(model_dist.log_prob(X))

    # M step

    # Sum the responsibilities over the dataset. 
    # This is the expected number of datapoints assigned to cluster k
    # shape [k]
    log_ns = jscipy.special.logsumexp(resps, axis=0)

    # Compute the new log weights, which is the normalized version of log_ns.
    # This is the probability that a given data point 
    # will be assigned to component k.
    # shape [k]
    log_weights = log_ns - jnp.log(n)

    # Compute new centroids
    # new centroids are sum of the data points weighted by the normalized responsibilities.
    # mu_k = (sum_i x_i p(z_i=k|x=x_i,theta)) / (sum_i p(z_i=k | x=x_i, theta))
    # shape [k, d]
    centroids = jnp.sum(
        (X[:, jnp.newaxis, :] * jnp.exp(resps)[:, :, jnp.newaxis]) /
        jnp.exp(log_ns[jnp.newaxis, :, jnp.newaxis]),
        axis=0)
    
    # Compute the data with the estimated centroids subtracted.
    # shape [n, k, d]
    centered_x = X[:, jnp.newaxis, :] - centroids[jnp.newaxis, :, :]

    # Compute the empirical scatter matrix
    # [n, k, d, d]
    outers = jnp.einsum('...i,...j->...ij', centered_x, centered_x)
    weighted_outers = outers * jnp.exp(resps[Ellipsis, jnp.newaxis,
                                                       jnp.newaxis])
    
    # Compute covariances as the weighted empirical scatter matrices.
    covs = jnp.sum(
        weighted_outers, axis=0) / jnp.exp(log_ns[:, jnp.newaxis, jnp.newaxis])
    covs = covs + jnp.eye(data_dim)*regularization
    all_resps = jax.ops.index_update(all_resps, t, resps)
    return (t+1, centroids, covs, log_weights, resps, new_elbo, elbo, all_resps)

  def em_pred(state):
    t, _, _, _, _, new_elbo, elbo, _ = state
    return jnp.logical_or(jnp.logical_and(new_elbo - elbo > tol, t < T), t <= 1)


  num_steps, out_mus, out_covs, out_log_ws, final_resps, final_elbo, _, all_resps = jax.lax.while_loop(
      em_pred, em_step, 
      (0, init_centroids, init_covs, init_log_weights, 
       jnp.zeros([n, k]), 0, -jnp.inf, init_all_resps)
      )
  return (out_mus, out_covs, out_log_ws), num_steps, all_resps, final_elbo, final_resps


def kmeans_pp_init(X, k, key):
  keys = jax.random.split(key, num=k)
  n, d = X.shape
  centroids = jnp.ones([k, d]) * jnp.inf
  centroids = jax.ops.index_update(centroids, 0,
                                   X[jax.random.randint(keys[0], [], 0, n), :])
  dist = lambda x, y: jnp.linalg.norm(x - y, axis=0, keepdims=False)

  def for_body(i, centroids):
    dists = vmap(vmap(dist, in_axes=(None, 0)), in_axes=(0, None))(X, centroids)
    min_square_dists = jnp.square(jnp.min(dists, axis=1))
    new_centroid_ind = jax.random.categorical(keys[i],
                                              jnp.log(min_square_dists))
    centroids = jax.ops.index_update(centroids, i, X[new_centroid_ind, :])
    return centroids

  return jax.lax.fori_loop(1, k, for_body, centroids)


def em_accuracy(xs, cs, k, key, num_iterations=25):
  _, _, em_log_membership_weights = em(xs, k, num_iterations, key)
  em_predicted_cs = jnp.argmax(em_log_membership_weights, axis=1)
  return util.permutation_invariant_accuracy(em_predicted_cs, cs, k)


def em_map(xs, k, key, num_iterations=25):
  """Computes EM's MAP estimate of cluster assignments."""
  _, _, em_log_membership_weights = em(xs, k, num_iterations, key)
  return jnp.argmax(em_log_membership_weights, axis=1)


def em_pairwise_metrics(xs, cs, k, key, num_iterations=25):
  _, _, em_log_membership_weights = em(xs, k, num_iterations, key)
  em_predicted_cs = jnp.argmax(em_log_membership_weights, axis=1)
  em_pairwise_preds = util.to_pairwise_preds(em_predicted_cs)
  true_pairwise_cs = util.to_pairwise_preds(cs)
  acc = jnp.mean(em_pairwise_preds == true_pairwise_cs)
  f1 = util.binary_f1(em_pairwise_preds, true_pairwise_cs)
  return acc, f1
