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

"""Functions for sampling batches of GMM data."""
from functools import partial

import util

import jax
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jscipy

def sample_spaced_means(key, num_means, covs, dist_mult, data_dim, max_rejections=1000):

  # The axes of an ellipsoid defined by x^TAx = 1 with A symmetric PSD
  # are along the eigenvectors of A with length equal to 1/sqrt(eigenvalue)

  # We are given the covariance matrix of a Gaussian and wish to compute a
  # confidence interval. The p-percent confidence interval for a N(0, Sigma)
  # Gaussian is
  #     x^T Sigma^-1 x <= chisq_invcdf(d,p).
  # where chisq_invcdf(d, p) is the inverse CDF of a Chi squared RV with d DOF.
  # For now let the chisq_invcdf just be a constant c. Then the ellipsoid we
  # wish to compute is
  #    x^T Sigma^-1 (1/c) x = 1.
  # Thus axis lengths of this ellipsoid will be 1/sqrt(eig(Sigma^-1 * (1/c))).
  # We are not given Sigma^-1 but do have Sigma, so to compute
  # eig(Sigma^-1 * (1/c)) we can first compute the eigenvalues of Sigma and then
  # obtain the eigenvalues of Sigma^-1 by taking 1/eig(Sigma). Then, we can
  # divide the eigenvalues of Sigma^-1 by c to obtain eig(Sigma^-1 *(1/c)).
  # Finally, we take the maximum over all eigenvalues of 1/sqrt(eig(Sigma^-1 * (1/c)))
  # and uses that as the radius of a ball around the mean.

  cov_eigs = vmap(jnp.linalg.eigvalsh)(covs)
  inv_cov_eigs = 1./cov_eigs
  norm_inv_cov_eigs = inv_cov_eigs / dist_mult
  radii = 1./jnp.sqrt(norm_inv_cov_eigs)
  max_radii = jnp.max(radii, axis=-1)
  thresholds = max_radii[:,None] + max_radii[None, :]

  bounds_length = jnp.power(100*num_means, 1./data_dim)
  bounds = (jnp.zeros([data_dim]) - jnp.ones([data_dim])*(bounds_length/2),
            jnp.zeros([data_dim]) + jnp.ones([data_dim])*(bounds_length/2))

  dist = lambda x, y: jnp.linalg.norm(x-y)

  def add_mean(i, state):
    means, num_rejections, key = state
    mask = jnp.arange(num_means) < i

    def spaced_mean_body(state):
      key, means, mask, unused_sample, num_rejections = state
      key, subkey = jax.random.split(key)
      sample = jax.random.uniform(subkey, shape=[data_dim],
                                  minval=bounds[0], maxval=bounds[1])
      return key, means, mask, sample, num_rejections + 1

    def spaced_mean_cond(state):
      unused_key, means, mask, sample, num_rejections = state

      dists = vmap(dist, in_axes=(0, None))(means, sample)
      too_close = (dists <= thresholds[i])
      return jnp.logical_and(
          jnp.any(jnp.logical_and(too_close, mask)),
          num_rejections <= max_rejections)

    key, subkey = jax.random.split(key)
    mean0 = jax.random.uniform(subkey, shape=[data_dim],
                               minval=bounds[0], maxval=bounds[1])

    key, subkey = jax.random.split(key)
    _, _, _, new_mean, new_rejs = jax.lax.while_loop(
        spaced_mean_cond, spaced_mean_body, (subkey, means, mask, mean0, 0))

    means = jax.ops.index_update(means, i, new_mean)
    num_rejections = jax.ops.index_update(num_rejections, i, new_rejs)
    return (means, num_rejections, key)

  means = jnp.zeros([num_means, data_dim])
  key, subkey = jax.random.split(key)
  means = jax.ops.index_update(
      means, 0, jax.random.uniform(subkey, shape=[data_dim],
                                   minval=bounds[0], maxval=bounds[1]))

  means, rejections, _ = jax.lax.fori_loop(
      1, num_means, add_mean, (means, jnp.zeros(num_means), key))
  return means


def sample_wishart(key, dof, scale):
  """Samples from the Wishart distribution.

  Args:
    key: A JAX PRNG key.
    dof: The degrees of freedom.
    scale: The scale of the distribution, a [d,d] PSD matrix.

  Returns:
    A sample from W(dof, scale)
  """
  data_dim = scale.shape[0]
  xs = jax.random.multivariate_normal(key,
                                      mean=jnp.zeros([data_dim]),
                                      cov=scale,
                                      shape=[dof])
  return jnp.einsum("ki,kj->ij", xs, xs)


def sample_scaled_wishart(key, dof, scale):
  """Samples W from a scaled wishart distribution with E[W] = scale.

  Args:
    key: A JAX PRNG key.
    dof: The degrees of freedom, higher means the distribution is more
      concentrated about scale.
    scale: The expected value of the distribution, a PSD [d,d] matrix.

  Returns:
    A sample from W(dof, (1/(dof-d-1))*scale)
  """
  data_dim = scale.shape[0]
  return sample_wishart(key, dof, (1./(dof-data_dim-1))*scale)


def sample_scaled_inverse_wishart(key, dof, scale):
  """Samples W from a scaled inverse wishart distribution with E[W^-1] = scale.

  Args:
    key: A JAX PRNG key.
    dof: The degrees of freedom, higher means the distribution is more
      concentrated about scale.
    scale: The expected value of the distribution, a PSD [d,d] matrix.

  Returns:
    A sample from IW(dof, (1/(dof-d-1))*scale)
  """
  data_dim = scale.shape[0]
  return jnp.linalg.inv(sample_wishart(key, dof, (1./(dof-data_dim-1))*scale))


def sample_gmm(key, mus, covs, w_logits, num_samples):
  """Samples from a Gaussian mixture model.

  Samples from a GMM with mixture component means 'mus', covariance matrices
  equal to I*scale, and mixing weights 'ws'.

  Args:
    key: JAX PRNG key.
    mus: A [K, D] set of K mixture component means.
    covs: A [K, D, D] set of covariance matrices of the mixture components.
    w_logits: A vector of [K] mixture component weight logits, need not sum to
      one. ws will be put through a softmax before being used as mixture
      weights.
    num_samples: The number of samples to draw.

  Returns:
    xs: A set of [num_samples, D] xs sampled from the GMM.
    cs: A set of [num_samples] integers in [0,K-1], the cluster assignments for
    each x.
  """
  subkey1, subkey2 = jax.random.split(key, num=2)
  cs = jax.random.categorical(subkey1, w_logits, shape=(num_samples,))
  x_mus = mus[cs]
  x_covs = covs[cs]
  xs = jax.random.multivariate_normal(subkey2, mean=x_mus, cov=x_covs)
  return xs, cs


def sample_masked_gmm(key, k, max_k, max_num_data_points, params):
  """Samples from a Gaussian mixture model, masked for performance.

  Args:
    key: A JAX PRNG key.
    k: The number of modes.
    max_k: An upper bound on the number of modes. Used for determining the
      shapes of the outputs.
    max_num_data_points: An upper bound on the number of data points, used
      to determine the shape of the outputs.
    params: The parameters of the GMM, a [max_k, data_dim] tensor of mus,
      a [max_k, data_dim, data_dim] tensor of mode covariances, and a
      [max_k] vector of log mixture weights.
  Returns:
    xs: A [max_num_data_points, data_dim] set of samples from the GMM.
      Only the first num_data_points entries are guaranteed to be valid samples.
    cs: A [max_num_data_points] set of cluster assignments for the sampled data
      points. Only the first num_data_points entries are guaranteed to be valid.
  """
  # log_ws is [max_k]
  # mus is [max_k, data_dim]
  # mode_scales is [max_k]
  mus, mode_scales, log_ws = params
  log_ws = jnp.where(jnp.arange(max_k) < k,
                     log_ws,
                     jnp.ones_like(log_ws)*-jnp.inf)
  xs, cs = sample_gmm(key, mus, mode_scales, log_ws, max_num_data_points)
  return xs, cs


def sample_all_gmm_params(key, k, max_k, data_dim, cov_dof, cov_shape, cov_prior, dist_mult):
  """Samples the parameters for a Gaussian mixture model.

  Samples GMM parameters from a prior. The means are sampled from a 'spaced'
  prior which is uniform but guaranteed to be at least a certain distance apart.
  The covariances are sampled from a scaled Wishart, i.e.
  W(cov_dof, (1/(cov_dof-data_dim=1)*cov_shape)). The logits of the mixture
  weights are sampled IID from a N(0,1).

  Args:
    key: A JAX PRNG Key.
    k: The number of modes.
    max_k: An upper bound on the number of modes. Used for determining the
      shapes of the outputs.
    data_dim: The dimensionality of the data.
    cov_dof: The degrees of freedom used in the Wishart prior for the covariance
      matrices.
    cov_shape: A [data_dim, data_dim] PSD matrix used to compute the shape prior
      for the covariance matrices.
    cov_prior: Either "wishart" or "inv_wishart", the prior to use for the covariance matrix.
    dist_mult: The multiplier for how separated the modes should be.
  Returns:
    mus: A [max_k, data_dim] set of mixture mode means. Only the first k entries
      are guaranteed to be valid.
    covs: A [max_k, data_dim, data_dim] set of mixture mode covariances. Only
      the first k entries are guaranteed to be valid.
    log_ws: A [max_k] vector containing the logits of the mixture weights. The
      true mixture weights are in the first k entries.
  """
  key1, key2, key3 = jax.random.split(key, num=3)
  if cov_prior == "inv_wishart":
    covs = vmap(sample_scaled_inverse_wishart, in_axes=(0, None, None))(
        jax.random.split(key1, num=max_k), cov_dof, cov_shape)
  else:
    covs = vmap(sample_scaled_wishart, in_axes=(0, None, None))(
        jax.random.split(key1, num=max_k), cov_dof, cov_shape)
  trace = jnp.sum(vmap(jnp.diag)(covs), axis=-1)
  mus = sample_spaced_means(key2, max_k, covs, dist_mult, data_dim)
  raw_ws = jax.random.normal(key3, shape=[max_k])*0.5
  cond = jnp.arange(max_k) < k
  log_ws = jnp.where(cond,
                     raw_ws,
                     jnp.ones_like(raw_ws)*-jnp.inf)
  return mus, covs, log_ws


def sample_gmm_mu_and_cov(key, k, max_k, data_dim, cov_dof, cov_shape, cov_prior,
                          log_weights, dist_mult):
  """Samples the parameters for a Gaussian mixture model with a wishart prior.

  Samples GMM parameters from a prior. The means are sampled from a 'spaced'
  prior which is uniform but guaranteed to be at least a certain distance apart.
  The covariances are sampled from a scaled Wishart, i.e.
  W(cov_dof, (1/(cov_dof-data_dim=1)*cov_shape)). The logits of the mixture
  weights are computed from the provided log_weights.

  Args:
    key: A JAX PRNG Key.
    k: The number of modes.
    max_k: An upper bound on the number of modes. Used for determining the
      shapes of the outputs.
    data_dim: The dimensionality of the data.
    cov_dof: The degrees of freedom used in the Wishart prior for the covariance
      matrices.
    cov_shape: A [data_dim, data_dim] PSD matrix used to compute the shape prior
      for the covariance matrices.
    log_weights: A [max_k] set of log mixture weights.
    dist_mult: The multiplier for how separated the modes should be.
  Returns:
    mus: A [max_k, data_dim] set of mixture mode means. Only the first k entries
      are guaranteed to be valid.
    covs: A [max_k, data_dim, data_dim] set of mixture mode covariances. Only
      the first k entries are guaranteed to be valid.
    log_ws: A [max_k] vector containing the logits of the mixture weights. The
      true mixture weights are in the first k entries.
  """
  key1, key2 = jax.random.split(key)
  if cov_prior == "inv_wishart":
    covs = vmap(sample_scaled_inverse_wishart, in_axes=(0, None, None))(
        jax.random.split(key1, num=max_k), cov_dof, cov_shape)
  else:
    covs = vmap(sample_scaled_wishart, in_axes=(0, None, None))(
        jax.random.split(key1, num=max_k), cov_dof, cov_shape)

  max_diag = jnp.amax(vmap(jnp.diag)(covs))
  mus = sample_spaced_means(key2, max_k, covs, dist_mult, data_dim)
  cond = jnp.arange(max_k) < k
  log_ws = jnp.where(cond,
                     log_weights,
                     jnp.ones_like(log_weights)*-jnp.inf)
  return mus, covs, log_ws


def sample_gmm_mu(key, k, max_k, data_dim, cov, log_weights, dist_mult):
  """Samples the means for a Gaussian mixture model.

  Samples GMM parameters from a prior. The means are sampled from a 'spaced'
  prior which is uniform but guaranteed to be at least a certain distance apart.
  The covariances and weights are computed from cov and log_weights.

  Args:
    key: A JAX PRNG Key.
    k: The number of modes.
    max_k: An upper bound on the number of modes. Used for determining the
      shapes of the outputs.
    data_dim: The dimensionality of the data.
    cov: A [data_dim, data_dim] covariance matrix for the data, will be the same
      for each mode.
    log_weights: A [max_k] set of log mixture weights.
    dist_mult: The multiplier for how separated the modes should be.
  Returns:
    mus: A [max_k, data_dim] set of mixture mode means. Only the first k entries
      are guaranteed to be valid.
    covs: A [max_k, data_dim, data_dim] set of mixture mode covariances. Only
      the first k entries are guaranteed to be valid.
    log_ws: A [max_k] vector containing the logits of the mixture weights. The
      true mixture weights are in the first k entries.
  """
  covs = jnp.tile(cov[jnp.newaxis, :, :], [max_k, 1, 1])
  max_diag = jnp.amax(jnp.diag(cov))
  mus = sample_spaced_means(key, max_k, covs, dist_mult, data_dim)
  cond = jnp.arange(max_k) < k
  log_ws = jnp.where(cond,
                     log_weights,
                     jnp.ones_like(log_weights)*-jnp.inf)
  return mus, covs, log_ws


def sample_random_gmm(key, k, max_k, max_num_data_points,
                      data_dim, sample_params_fn):
  """Samples data from a GMM with random parameters.

  Args:
    key: A JAX PRNG Key.
    k: The number of modes.
    max_k: An upper bound on the number of modes. Used for determining the
      shapes of the outputs.
    max_num_data_points: An upper bound on the number of data points, used
      to determine the shape of the outputs.
    data_dim: The dimensionality of the data.
    sample_params_fn: A function which returns a sample of the parameters of a
      GMM. Must accept a key, num_modes, max_num_modes, and data_dim and return
      A three-tuple of (mus, covs, and log_ws).
  Returns:
    xs: A [max_num_data_points, data_dim] set of samples from the GMM.
      Only the first num_data_points entries are guaranteed to be valid samples.
    cs: A [max_num_data_points] set of cluster assignments for the sampled data
      points. Only the first num_data_points entries are guaranteed to be valid.
    params: A tuple of (mus, covs, log_ws).
      mus: A [max_k, data_dim] set of mixture mode means. Only the first k
        entries are guaranteed to be valid.
      covs: A [max_k, data_dim, data_dim] set of mixture mode covariances. Only
        the first k entries are guaranteed to be valid.
      log_ws: A [max_k] vector containing the logits of the mixture weights. The
        true mixture weights are in the first k entries.
  """
  key1, key2 = jax.random.split(key)
  params = sample_params_fn(key1, k, max_k, data_dim)
  xs, cs = sample_masked_gmm(key2, k, max_k, max_num_data_points, params)
  return xs, cs, params


def sample_random_gmm_batch(
    key, ks, max_k, max_num_data_points, data_dim, sample_params_fn):
  """Samples batches of data from GMMs with random parameters.

  Args:
    key: A JAX PRNG Key.
    ks: A [batch_size] tensor, the number of modes for each GMM in the batch.
    max_k: An upper bound on the number of modes. Used for determining the
      shapes of the outputs.
    max_num_data_points: An upper bound on the number of data points, used
      to determine the shape of the outputs.
    data_dim: The dimensionality of the data.
    sample_params_fn: A function which returns a sample of the parameters of a
      GMM. Must accept a key, num_modes, max_num_modes, and data_dim and return
      A three-tuple of (mus, covs, and log_ws).
  Returns:
    xs: A [batch_size, max_num_data_points, data_dim] tensor of samples from the
      GMMs. Only the first num_data_points[i] entries are guaranteed to be valid
      samples for the ith batch element.
    cs: A [batch_size, max_num_data_points] set of cluster assignments for the
      sampled data points. Only the first num_data_points[i] entries are
      guaranteed to be valid for the ith batch element.
    params: A tuple of (mus, covs, log_ws).
      mus: A [batch_size, max_k, data_dim] set of mixture mode means. Only the
        first ks[i] means are guaranteed to be valid for the ith batch element.
      covs: A [batch_size, max_k, data_dim, data_dim] set of mixture mode
        covariances. Only the first k[i] entries are guaranteed to be valid for
        the ith batch element.
      log_ws: A [batch_size, max_k] vector containing the logits of the
        mixture weights. The true mixture weights for the ith batch element
        are contained in the first k[i] entries.
  """
  key1, key2 = jax.random.split(key)
  params = vmap(
      sample_params_fn,
      in_axes=(0, 0, None, None))(
          jax.random.split(key1, num=ks.shape[0]), ks, max_k, data_dim)
  xs, cs = vmap(
      sample_masked_gmm,
      in_axes=(0, 0, None, None, (0, 0, 0)))(
          jax.random.split(key2, num=ks.shape[0]), ks, max_k,
          max_num_data_points, params)
  return xs, cs, params


def sample_random_gmm_batch_with_k_range(
    key, batch_size, min_k, max_k, max_num_data_points, data_dim,
    sample_params_fn):
  """Sample batches of data from GMMs with num modes in a range of ks."""

  k1, k2 = jax.random.split(key)
  ks = jax.random.choice(k1, jnp.arange(min_k, stop=max_k+1),
                         shape=(batch_size,), replace=True)
  xs, cs, params = sample_random_gmm_batch(
      k2, ks, max_k, max_num_data_points,
      data_dim, sample_params_fn)
  return xs, cs, ks, params


def batch_with_random_mu_fixed_ks(key, ks, max_k, max_num_data_points, data_dim,
                                  cov, log_ws, dist_mult):

  def sample_params_fn(key, k, max_k, data_dim):
    return sample_gmm_mu(key, k, max_k, data_dim, cov, log_ws, dist_mult)

  return sample_random_gmm_batch(key, ks, max_k, max_num_data_points, data_dim,
                                 sample_params_fn)


def batch_with_random_mu_random_ks(key, batch_size, min_k, max_k,
                                   max_num_data_points, data_dim, cov, log_ws,
                                   dist_mult):
  k1, k2 = jax.random.split(key)
  ks = jax.random.choice(
      k1, jnp.arange(min_k, stop=max_k + 1), shape=(batch_size,), replace=True)
  xs, cs, params = batch_with_random_mu_fixed_ks(k2, ks, max_k,
                                                 max_num_data_points, data_dim,
                                                 cov, log_ws, dist_mult)
  return xs, cs, ks, params


def batch_with_random_mu_cov_fixed_ks(key, ks, max_k, max_num_data_points,
                                      data_dim, cov_dof, cov_shape, cov_prior, log_ws,
                                      dist_mult):

  def sample_params_fn(key, k, max_k, data_dim):
    return sample_gmm_mu_and_cov(key, k, max_k, data_dim, cov_dof, cov_shape, cov_prior,
                                 log_ws, dist_mult)
  xs, cs, params = sample_random_gmm_batch(
      key, ks, max_k, max_num_data_points, data_dim, sample_params_fn)

  scales = vmap(vmap(jnp.linalg.cholesky))(params[1])
  params = (params[0], scales, params[2])
  return xs, cs, params


def batch_with_random_mu_cov_random_ks(
    key, batch_size, min_k, max_k, max_num_data_points, data_dim, cov_dof,
    cov_shape, cov_prior, log_ws, dist_mult):
  k1, k2 = jax.random.split(key)
  ks = jax.random.choice(
      k1, jnp.arange(min_k, stop=max_k + 1), shape=(batch_size,), replace=True)

  xs, cs, params = batch_with_random_mu_cov_fixed_ks(
          k2, ks, max_k, max_num_data_points, data_dim, 
          cov_dof, cov_shape, cov_prior, log_ws, dist_mult)

  return xs, cs, ks, params


def batch_with_all_random_params_fixed_ks(key, ks, max_k, max_num_data_points,
                                          data_dim, cov_dof, cov_shape, cov_prior,
                                          dist_mult):

  def sample_params_fn(key, k, max_k, data_dim):
    return sample_all_gmm_params(key, k, max_k, data_dim, cov_dof, cov_shape, cov_prior,
                                 dist_mult)

  xs, cs, params = sample_random_gmm_batch(key, ks, max_k, max_num_data_points,
                                           data_dim, sample_params_fn)

  scales = vmap(vmap(jnp.linalg.cholesky))(params[1])
  params = (params[0], scales, params[2])
  return xs, cs, params


def batch_with_all_random_params_random_ks(
    key, batch_size, min_k, max_k, max_num_data_points, data_dim, cov_dof,
    cov_shape, cov_prior, dist_mult):
  k1, k2 = jax.random.split(key)
  ks = jax.random.choice(k1, jnp.arange(min_k, stop=max_k+1),
                         shape=(batch_size,), replace=True)

  xs, cs, params = batch_with_all_random_params_fixed_ks(
      k2, ks, max_k, max_num_data_points, data_dim, cov_dof, cov_shape, cov_prior,
      dist_mult)

  return xs, cs, ks, params


def sample_batch_random_ks(
    key, sampling_type, batch_size, min_k, max_k, data_points_per_mode,
    data_dim, mode_variance, cov_dof, cov_prior, dist_mult):
  if sampling_type == "mean":
    xs, cs, ks, params = batch_with_random_mu_random_ks(
        key, batch_size, min_k, max_k,
        max_k*data_points_per_mode, data_dim,
        jnp.eye(data_dim)*mode_variance, jnp.zeros([max_k]), dist_mult)
  elif sampling_type == "mean_scale":
    xs, cs, ks, params = batch_with_random_mu_cov_random_ks(
        key, batch_size, min_k, max_k,
        max_k*data_points_per_mode, data_dim,
        cov_dof, jnp.eye(data_dim), cov_prior, jnp.zeros([max_k]), dist_mult)
  elif sampling_type == "mean_scale_weight":
    xs, cs, ks, params = batch_with_all_random_params_random_ks(
        key, batch_size, min_k, max_k,
        max_k*data_points_per_mode, data_dim,
        cov_dof, jnp.eye(data_dim), cov_prior, dist_mult)
  return xs, cs, ks, params


def sample_batch_fixed_ks(
    key, sampling_type, ks, max_k, data_points_per_mode, data_dim,
    mode_variance, cov_dof, cov_prior, dist_mult):
  if sampling_type == "mean":
    xs, cs, params = batch_with_random_mu_fixed_ks(
        key, ks, max_k, max_k*data_points_per_mode, data_dim,
        jnp.eye(data_dim)*mode_variance, jnp.zeros([max_k]), dist_mult)
  elif sampling_type == "mean_scale":
    xs, cs, params = batch_with_random_mu_cov_fixed_ks(
        key, ks, max_k, max_k*data_points_per_mode, data_dim,
        cov_dof, jnp.eye(data_dim), cov_prior, jnp.zeros([max_k]), dist_mult)
  elif sampling_type == "mean_scale_weight":
    xs, cs, params = batch_with_all_random_params_fixed_ks(
        key, ks, max_k, max_k*data_points_per_mode, data_dim,
        cov_dof, jnp.eye(data_dim), cov_prior, dist_mult)
  return xs, cs, params


def sample_batch_fixed_ks2(
    key, sampling_type, ks, max_k, max_num_data_points, data_dim,
    mode_variance, cov_dof, cov_prior, dist_mult):
  if sampling_type == "mean":
    xs, cs, params = batch_with_random_mu_fixed_ks(
        key, ks, max_k, max_num_data_points, data_dim,
        jnp.eye(data_dim)*mode_variance, jnp.zeros([max_k]), dist_mult)
  elif sampling_type == "mean_scale":
    xs, cs, params = batch_with_random_mu_cov_fixed_ks(
        key, ks, max_k, max_num_data_points, data_dim,
        cov_dof, jnp.eye(data_dim), cov_prior, jnp.zeros([max_k]), dist_mult)
  elif sampling_type == "mean_scale_weight":
    xs, cs, params = batch_with_all_random_params_fixed_ks(
        key, ks, max_k, max_num_data_points, data_dim,
        cov_dof, jnp.eye(data_dim), cov_prior, dist_mult)
  return xs, cs, params


@partial(jax.jit, static_argnums=5)
def sample_banana_mm(key, mus, scale, r, w_logits, num_samples):
  """Samples from a mixture of bananas model.

  Args:
    key: JAX PRNG key.
    mus: A [K, D] set of K mixture component means.
    scale: A scalar float value for the scale of the gaussian noise
      (width of bananas).
    r: A scalar float value for the radius of the bananas.
    w_logits: A vector of [K] mixture component weight logits.
      ws need not sum to one, and will be put through a softmax before being
      used as mixture weights.
    num_samples: The number of samples to draw.

  Returns:
    xs: A set of [num_samples, D] xs sampled from the MoBM.
    cs: A set of [num_samples] integers in [0,K-1], the cluster assignments for
      each x.
  """
  keys = jax.random.split(key, num=4)
  cs = jax.random.categorical(keys[0], w_logits, shape=(num_samples,))
  x_mus = mus[cs]
  cluster_angles = jax.random.uniform(
      keys[1], shape=(num_samples,)) * 2 * jnp.pi
  x_angles = cluster_angles[cs] + jax.random.uniform(
      keys[2], shape=(num_samples,)) * jnp.pi
  x_coord = jnp.sin(x_angles)
  y_coord = jnp.cos(x_angles)
  r_noise = scale*jax.random.normal(keys[3], shape=[num_samples])
  xs = x_mus + (jnp.stack([x_coord, y_coord]).T)*((r + r_noise)[:, jnp.newaxis])
  return xs, cs


@partial(jax.jit)
def log_joint(xs, cs, mus, scale, ws):
  """Evaluates the log joint probability of a GMM.

  The GMM is defined by the following sampling process:

  c_i ~ Categorical(w) for i=1,...,N
  x_i ~ Normal(mus[c_i], scale^2) for i=1,...,N

  Args:
    xs: A set of [N, D] values to compute the log probability of.
    cs: A shape [N] integer vector, the cluster assignments for each x.
    mus: A set of [K, D] mixture component means.
    scale: A scalar float, the scale of the mixture components.
    ws: A vector of shape [K], the mixture weights of the GMM. Need not be
      normalized.

  Returns:
    A [N] float vector, the log probabilities of each X.
  """
  log_p_c = jax.vmap(util.categorical_logpmf, in_axes=(0, None))(cs, ws)
  log_p_x = jnp.sum(
      jscipy.stats.norm.logpdf(xs, loc=mus[cs], scale=scale), axis=1)
  return log_p_c + log_p_x


@partial(jax.jit)
def log_joint_with_prior(xs, cs, mus, scale, ws, mu_prior_mean, mu_prior_scale,
                         unused_w_prior_conc):
  """Evaluates the log joint probability of a GMM with a prior on some of the parameters.

  The GMM is defined by the following sampling process:

  w ~ Dirichlet(w_prior_conc)
  mu_i ~ Normal(mu_prior_mean, mu_prior_scale^2) for i=1,...,K
  c_i ~ Categorical(w) for i=1,...,N
  x_i ~ Normal(mus[c_i], scale^2) for i=1,...,N

  Args:
    xs: A set of [N, D] values to compute the log probability of.
    cs: A shape [N] integer vector, the cluster assignments for each x.
    mus: A set of [K, D] mixture component means.
    scale: A scalar float, the scale of the mixture components.
    ws: A vector of shape [K], the mixture weights of the GMM. Need not be
      normalized.
    mu_prior_mean: The prior mean of the mixture component means.
    mu_prior_scale: The prior scale of the mixture component means.
    unused_w_prior_conc: The concentration parameter for the Dirichlet prior on
      w.

  Returns:
    A scalar float, the log probability of the data.
  """
  log_p = jnp.sum(log_joint(xs, cs, mus, scale, ws))
  log_p_mu = jnp.sum(
      jscipy.stats.norm.logpdf(mus, loc=mu_prior_mean, scale=mu_prior_scale))
  log_p_w = 0.  # jnp.sum(scipy.stats.dirichlet.logpdf(ws, w_prior_conc))
  return log_p + log_p_mu + log_p_w


@partial(jax.jit)
def marginal(x, mus, scale, ws):
  """Computes the marginal probability of x under a GMM.

  Args:
    x: A shape [D] vector, the data point to compute p(x) for.
    mus: A [K,D] matrix, the K D-dimensional mixture component means.
    scale: The scale of the mixture components
    ws: A shape [K] vector, the mixture weights.

  Returns:
    p(x), a float scalar.
  """
  cs = jnp.arange(mus.shape[0])[:, jnp.newaxis]
  x = x[jnp.newaxis, :]
  log_ps = jax.vmap(
      log_joint, in_axes=(None, 0, None, None, None))(x, cs, mus, scale, ws)
  return jnp.exp(jscipy.special.logsumexp(log_ps))


batch_marginal = jax.jit(jax.vmap(marginal, in_axes=(0, None, None, None)))
