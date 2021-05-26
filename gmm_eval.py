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
from functools import partial
import itertools

import jax
from jax import vmap
import jax.numpy as jnp
import numpy as onp
import sklearn
import sklearn.cluster
import sklearn.metrics
import sklearn.mixture

import gmm_models
import em

import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

def log_marginal(xs, mus, covs, log_ws):
  """Computes the log_marginal probability of x under a GMM.

  Args:
    xs: A shape [N, D] vector, the data to compute log p(x) for.
    mus: A [K,D] matrix, the K D-dimensional mixture component means.
    covs: A [K,D,D] matrix, the covariances of the mixture components.
    log_ws: A shape [K] vector, the log mixture weights.

  Returns:
    p(x), a tensor of shape [N].
  """
  return tfd.MixtureSameFamily(
    tfd.Categorical(logits=log_ws),
    tfd.MultivariateNormalFullCovariance(loc=mus, covariance_matrix=covs)).log_prob(xs)  

def responsibilities(xs, mus, covs, log_ws):
  c_dist = tfd.MultivariateNormalFullCovariance(loc=mus, covariance_matrix=covs)
  # [num_data_points, num_components]
  comp_lps = vmap(c_dist.log_prob)(xs)
  # [num_data_points, num_components]
  resps = comp_lps + log_ws[jnp.newaxis,...] - log_marginal(xs, mus, covs, log_ws)[:, jnp.newaxis]
  return resps

def masked_log_marginal_per_x(xs, mus, covs, log_ws, mask, k):
  max_k = mus.shape[0]
  log_ws = jnp.where(jnp.arange(max_k) < k,
                     log_ws,
                     jnp.full_like(log_ws, -jnp.inf))
  log_marginals = log_marginal(xs, mus, covs, log_ws)
  masked_log_marginals = jnp.where(
      mask,
      log_marginals,
      jnp.zeros_like(log_marginals))
  return jnp.sum(masked_log_marginals) / jnp.sum(mask)

masked_log_marginal_per_x = jax.jit(masked_log_marginal_per_x)

def make_mask(true_cs, num_data_points):
  max_num_data_points = true_cs.shape[0]
  mask = jnp.logical_and(
      jnp.arange(max_num_data_points) < num_data_points,
      true_cs >= 0)
  return mask

def masked_binary_f1(preds, labels, mask):
  tp = jnp.sum(mask * jnp.logical_and(labels == 1, preds == 1))
  fp = jnp.sum(mask * jnp.logical_and(labels == 0, preds == 1))
  fn = jnp.sum(mask * jnp.logical_and(labels == 1, preds == 0))
  return tp/(tp + 0.5*(fp + fn))

def masked_accuracy(preds, labels, mask):
  return jnp.sum(mask * (preds == labels)) / jnp.sum(mask)

def to_pairwise(x):
  pairwise = x[jnp.newaxis, :] == x[:, jnp.newaxis]
  return pairwise[jnp.tril_indices_from(pairwise, k=-1)]

def make_pairwise(f):

  def pairwise_f(preds, labels, mask):
    pw_preds = to_pairwise(preds)
    pw_labels = to_pairwise(labels)
    mask = mask[jnp.newaxis,:] * mask[:, jnp.newaxis]
    mask = mask[jnp.tril_indices_from(mask, k=-1)]
    return f(pw_preds, pw_labels, mask)

  return pairwise_f

masked_pairwise_f1 = make_pairwise(masked_binary_f1)
masked_pairwise_accuracy = make_pairwise(masked_accuracy)

def classification_metrics(true_cs, pred_cs, num_points):
  mask = make_mask(true_cs, num_points)
  acc = masked_pairwise_accuracy(pred_cs, true_cs, mask)
  f1 = masked_pairwise_f1(pred_cs, true_cs, mask)
  return acc, f1, mask

def batch_metrics(xs, pred_params, true_cs, pred_cs, num_points, ks):
  accs, f1s, masks = vmap(classification_metrics)(true_cs, pred_cs, num_points)
  # Compute train marginal likelihood of predicted model 
  pred_mus, pred_covs, pred_log_ws = pred_params
  pred_lls = vmap(masked_log_marginal_per_x)(
      xs, pred_mus, pred_covs, pred_log_ws, masks, ks)
  return accs, f1s, pred_lls

def metrics(xs, pred_params, true_cs, pred_cs, num_points, ks):
  acc, f1, mask = classification_metrics(true_cs, pred_cs, num_points)
  pred_mus, pred_covs, pred_log_ws = pred_params
  pred_ll = masked_log_marginal_per_x(xs, pred_mus, pred_covs, pred_log_ws, mask, ks)
  return acc, f1, pred_ll

def em_fit_and_predict(xs, num_modes, prob_type, mode_var):
  data_dim = xs.shape[1]
  if prob_type == "mean_scale_weight":
    model = sklearn.mixture.GaussianMixture(
        n_components=num_modes,
        covariance_type="full",
        init_params="kmeans",
        n_init=1).fit(xs)
    covs = model.covariances_
  elif prob_type == "mean":
    model = sklearn.mixture.GaussianMixture(
        n_components=num_modes,
        covariance_type="spherical",
        init_params="kmeans",
        weights_init=onp.full([num_modes], 1./num_modes),
        precisions_init=onp.full([num_modes], 1./mode_var),
        n_init=1).fit(xs)
    covs = onp.array([l*onp.eye(data_dim) for l in model.covariances_])
  else:
    assert False, "Wrong problem type in em fit and predict"

  preds = model.predict(xs)
  mus = model.means_
  log_ws = onp.log(model.weights_)
  return preds, (mus, covs, log_ws)

def dpmm_fit_and_predict(xs, num_modes):
  model = sklearn.mixture.BayesianGaussianMixture(
      covariance_type="full",
      n_components=num_modes,
      init_params="kmeans").fit(xs)
  preds = model.predict(xs)
  mus = model.means_
  covs = model.covariances_
  log_ws = onp.log(model.weights_)
  return preds, (mus, covs, log_ws)

def spectral_rbf_fit_and_predict(xs, num_modes):
  return sklearn.cluster.SpectralClustering(
      n_clusters=num_modes,
      n_init=1,
      affinity="rbf").fit_predict(xs)


def agglomerative_fit_and_predict(xs, num_modes):
  return sklearn.cluster.AgglomerativeClustering(
      n_clusters=num_modes,
      affinity="euclidean").fit_predict(xs)

@partial(jax.jit, static_argnums=(5,))
def compute_baseline_metrics(
    key, train_xs, train_cs, test_xs, test_cs, num_modes):
  batch_size, num_data_points, _ = train_xs.shape
  batch_em = vmap(em.em, in_axes=(0, None, None, 0, None, None))
  em_tol = 1e-5
  em_reg = 1e-4
  em_max_num_steps = 200
  pred_params, _, _, _, train_resps = batch_em(
      train_xs, num_modes, em_max_num_steps, jax.random.split(key, num=batch_size), em_tol, em_reg)
  
  #[batch_size, num_data_points]
  train_pred_cs = jnp.argmax(train_resps, axis=-1)
  pred_mus, pred_covs, pred_log_ws = pred_params
  test_pred_cs = vmap(gmm_models.masked_classify_points, in_axes=(0,0,0,0,None))(
      test_xs, pred_mus, pred_covs, pred_log_ws, num_modes)

  train_accs, train_f1s, train_lls = vmap(metrics, in_axes=(0, 0, 0, 0, None, None))(
    train_xs, pred_params, train_cs, train_pred_cs, num_data_points, num_modes)
  test_accs, test_f1s, test_lls = vmap(metrics, in_axes=(0, 0, 0, 0, None, None))(
    test_xs, pred_params, test_cs, test_pred_cs, num_data_points, num_modes)
  return (train_accs, test_accs, train_f1s, 
      test_f1s, train_lls, test_lls)

def compute_masked_baseline_metrics(
    train_xs, train_cs, test_xs, test_cs, prob_type, mode_var, num_modes, num_points):

  batch_size = train_xs.shape[0]
  # EM
  em_train_acc_tot = 0.
  em_train_f1_tot = 0.
  em_train_ll_tot = 0.
  em_test_acc_tot = 0.
  em_test_f1_tot = 0.
  em_test_ll_tot = 0.
  for i in range(batch_size):
    n_i = num_points[i]
    k_i = num_modes[i]
    train_xi = train_xs[i, :n_i]
    train_ci = train_cs[i, :n_i]
    test_xi = test_xs[i, :n_i]
    test_ci = test_cs[i, :n_i]
    
    train_pred_cs, pred_params = em_fit_and_predict(train_xi, k_i, prob_type, mode_var)
    pred_mus, pred_covs, pred_log_ws = pred_params
    test_pred_cs = gmm_models.masked_classify_points(
        test_xi, pred_mus, pred_covs,  pred_log_ws, k_i)

    train_acc, train_f1, train_ll = metrics(
        train_xi, pred_params, train_ci, train_pred_cs, n_i, k_i)
    test_acc, test_f1, test_ll = metrics(
        test_xi, pred_params, test_ci, test_pred_cs, n_i, k_i)

    em_train_acc_tot += train_acc
    em_train_f1_tot += train_f1
    em_train_ll_tot += train_ll
    em_test_acc_tot += test_acc 
    em_test_f1_tot += test_f1
    em_test_ll_tot += test_ll

  em_train_acc = em_train_acc_tot / batch_size
  em_train_f1 = em_train_f1_tot / batch_size
  em_train_ll = em_train_ll_tot / batch_size
  em_test_acc = em_test_acc_tot / batch_size
  em_test_f1 = em_test_f1_tot / batch_size
  em_test_ll = em_test_ll_tot / batch_size

  return (em_train_acc, em_test_acc, em_train_f1, em_test_f1, em_train_ll, em_test_ll)


def masked_em_train_metrics(xs, cs, prob_type, mode_var, num_modes, num_points):
  batch_size = xs.shape[0]
  # EM
  em_acc_tot = 0.
  em_f1_tot = 0.
  em_ll_tot = 0.
  for i in range(batch_size):
    n_i = num_points[i]
    k_i = num_modes[i]
    xi = xs[i, :n_i]
    ci = cs[i, :n_i]
    
    pred_cs, pred_params = em_fit_and_predict(xi, k_i, prob_type, mode_var)

    acc, f1, ll = metrics(
        xi, pred_params, ci, pred_cs, n_i, k_i)

    em_acc_tot += acc
    em_f1_tot += f1
    em_ll_tot += ll

  em_acc = em_acc_tot / batch_size
  em_f1 = em_f1_tot / batch_size
  em_ll = em_ll_tot / batch_size
  return (em_acc, em_f1, em_ll)
