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

"""Functions for computing performance metrics of various inference methods.
"""
import collections
import functools
import itertools

import jax
from jax import vmap
import jax.numpy as jnp
import numpy as onp
import sklearn
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

def log_marginal(xs, mus, covs, log_ws):
  """Computes the log_marginal probability of x under a GMM.

  Args:
    x: A shape [N, D] vector, the data to compute log p(x) for.
    mus: A [K,D] matrix, the K D-dimensional mixture component means.
    covs: A [K,D,D] matrix, the covariances of the mixture components.
    log_ws: A shape [K] vector, the log mixture weights.

  Returns:
    p(x), a float scalar.
  """
  return tfd.MixtureSameFamily(
    tfd.Categorical(logits=log_ws),
    tfd.MultivariateNormalFullCovariance(loc=mus, covariance_matrix=covs)).log_prob(xs)  

def masked_log_marginal_per_x(xs, mus, covs, log_ws, num_points, k):
  max_num_points = xs.shape[0]
  max_k = mus.shape[0]
  log_ws = jnp.where(jnp.arange(max_k) < k,
                     log_ws,
                     jnp.full_like(log_ws, -jnp.inf))
  log_marginals = log_marginal(xs, mus, covs, log_ws)
  masked_log_marginals = jnp.where(jnp.arange(max_num_points) < num_points,
      log_marginals,
      jnp.zeros_like(log_marginals))
  return jnp.sum(masked_log_marginals) / num_points

masked_log_marginal_per_x = jax.jit(masked_log_marginal_per_x)

def masked_to_pairwise(x, num_points):
    pairwise = x[jnp.newaxis, :] == x[:, jnp.newaxis]
    pairwise = pairwise[jnp.tril_indices_from(pairwise, k=-1)]
    num_points = (num_points *(num_points - 1))/2
    return pairwise, num_points


def make_pairwise(f):
  
  def pairwise_f(preds, labels, num_points):
    pw_preds, pw_num_points = masked_to_pairwise(preds, num_points)
    pw_labels, _ = masked_to_pairwise(labels, num_points)
    return f(pw_preds, pw_labels, pw_num_points)

  return pairwise_f

def make_permutation_invariant(f):

  def permutation_invariant_f(preds, labels, num_points, k):
    perms = jnp.array(list(itertools.permutations(range(k))))
    vals = jax.lax.map(lambda p: f(preds, p[labels], num_points), perms)
    return jnp.max(vals)

  return permutation_invariant_f

def masked_accuracy(preds, labels, num_points):
  eq = (preds == labels)
  masked_eq = jnp.where(jnp.arange(preds.shape[0]) < num_points, eq, jnp.zeros_like(eq))
  return jnp.sum(masked_eq) / num_points

masked_pairwise_accuracy = jax.jit(make_pairwise(masked_accuracy))
masked_perm_invariant_accuracy = make_permutation_invariant(masked_accuracy)

def masked_binary_f1(preds, labels, num_points):
  dim = preds.shape[0]
  tp = jnp.sum(jnp.where(jnp.logical_and(
      jnp.arange(dim) < num_points,
      jnp.logical_and(labels == 1, preds == 1)),
      jnp.ones_like(preds), jnp.zeros_like(preds)))
  fp = jnp.sum(jnp.where(jnp.logical_and(
      jnp.arange(dim) < num_points,
      jnp.logical_and(labels == 0, preds == 1)),
      jnp.ones_like(preds), jnp.zeros_like(preds)))
  fn = jnp.sum(jnp.where(jnp.logical_and(
      jnp.arange(dim) < num_points,
      jnp.logical_and(labels == 1, preds == 0)),
      jnp.ones_like(preds), jnp.zeros_like(preds)))
  return tp/(tp + 0.5*(fp + fn))

masked_pairwise_binary_f1 = jax.jit(make_pairwise(masked_binary_f1))

def compute_classification_metrics(true_cs, pred_cs, num_points):
  accs = vmap(masked_pairwise_accuracy)(pred_cs, true_cs, num_points)
  f1s = vmap(masked_pairwise_binary_f1)(pred_cs, true_cs, num_points)
  return jnp.mean(accs), jnp.mean(f1s)

def compute_metrics(xs, pred_params, true_cs, pred_cs, num_points, ks):
  avg_acc, avg_f1 = compute_classification_metrics(true_cs, pred_cs, num_points)
  # Compute train marginal likelihood of predicted model 
  pred_mus, pred_covs, pred_log_ws = pred_params
  pred_log_marginal = vmap(masked_log_marginal_per_x)(
      xs, pred_mus, pred_covs, pred_log_ws, num_points, ks)
  return avg_acc, avg_f1, jnp.mean(pred_log_marginal)

def em_fit_and_predict(xs, num_modes):
  model = sklearn.mixture.GaussianMixture(
      n_components=num_modes,
      covariance_type="full",
      init_params="kmeans",
      n_init=1).fit(xs)
  preds = model.predict(xs)
  mus = model.means_
  covs = model.covariances_
  log_ws = onp.log(model.weights_)
  return preds, mus, covs, log_ws


def spectral_rbf_fit_and_predict(xs, num_modes):
  return sklearn.cluster.SpectralClustering(
      n_clusters=num_modes,
      n_init=1,
      affinity="rbf").fit_predict(xs)


def agglomerative_fit_and_predict(xs, num_modes):
  return sklearn.cluster.AgglomerativeClustering(
      n_clusters=num_modes,
      affinity="euclidean").fit_predict(xs)


def compute_masked_baseline_metrics(xs, cs, num_modes, num_points):
  batch_size = xs.shape[0]

  # EM
  em_acc_tot = 0.
  em_f1_tot = 0.
  em_ll_tot = 0.
  for i in range(batch_size):
    pred_cs, pred_mus, pred_covs, pred_log_ws = em_fit_and_predict(
        xs[i, :num_points[i]], num_modes[i])
    em_acc_tot += masked_pairwise_accuracy(pred_cs, cs[i, :num_points[i]], num_points[i])
    em_f1_tot += masked_pairwise_binary_f1(pred_cs, cs[i, :num_points[i]], num_points[i])
    em_ll_tot += masked_log_marginal_per_x(
        xs[i, :num_points[i]], pred_mus, pred_covs, pred_log_ws, num_points[i], num_modes[i])
  em_avg_acc = em_acc_tot / batch_size
  em_avg_f1 = em_f1_tot / batch_size
  em_avg_ll = em_ll_tot / batch_size
  
  #Spectral RBF
  srbf_acc_tot = 0.
  srbf_f1_tot = 0.
  for i in range(batch_size):
    pred_cs = spectral_rbf_fit_and_predict(xs[i, :num_points[i]], num_modes[i])
    srbf_acc_tot += masked_pairwise_accuracy(pred_cs, cs[i, :num_points[i]], num_points[i])
    srbf_f1_tot += masked_pairwise_binary_f1(pred_cs, cs[i, :num_points[i]], num_points[i])
  srbf_avg_acc = srbf_acc_tot / batch_size
  srbf_avg_f1 = srbf_f1_tot / batch_size

  #Agglomerative clustering
  agg_acc_tot = 0.
  agg_f1_tot = 0.
  for i in range(batch_size):
    pred_cs = agglomerative_fit_and_predict(xs[i, :num_points[i]], num_modes[i])
    agg_acc_tot += masked_pairwise_accuracy(pred_cs, cs[i, :num_points[i]], num_points[i])
    agg_f1_tot += masked_pairwise_binary_f1(pred_cs, cs[i, :num_points[i]], num_points[i])
  agg_avg_acc = agg_acc_tot / batch_size
  agg_avg_f1 = agg_f1_tot / batch_size

  return ((em_avg_acc, em_avg_f1, em_avg_ll), 
          (srbf_avg_acc, srbf_avg_f1), 
          (agg_avg_acc, agg_avg_f1))