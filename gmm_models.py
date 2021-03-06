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

"""Transformer models for performing inference in a GMM.
"""
from functools import partial

import transformer
import util

import flax
import jax
from jax import vmap
import jax.numpy as jnp
import jax.random
import jax.scipy as jscipy


class MeanInferenceMachine(object):
  """Model which predicts cluster means from a batch of data."""

  def __init__(self,
               data_dim=2,
               max_k=2,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               weight_init=jax.nn.initializers.xavier_uniform()):
    """Creates the model.

    Args:
      data_dim: The dimensionality of the data points to be fed in.
      max_k: The maximum number of clusters that could occur in the data.
      max_num_data_points: The maximum number of data points that could be
        fed in at one time.
      num_heads: The number of heads to use in the transformer.
      num_encoders: The number of encoder layers to use in the transformer.
      num_decoders: The number of decoder layers to use in the transformer.
      qkv_dim: The dimensions of the queries, keys, and values in the
        transformer.
      activation_fn: The activation function to use for hidden layers.
      weight_init: The weight initializer.
    """
    self.data_dim = data_dim
    self.max_k = max_k
    self.max_num_data_points = max_num_data_points
    self.tfmr = transformer.EncoderDecoderTransformer.partial(
        target_dim=data_dim,
        max_input_length=max_num_data_points, max_target_length=max_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        activation_fn=activation_fn, weight_init=weight_init)

  def init_params(self, key):
    """Initializes the parameters of the model using dummy data.

    Args:
      key: A JAX PRNG key
    Returns:
      params: The parameters of the model.
    """
    batch_size = 1
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(
        subkey, [batch_size, self.max_num_data_points, self.data_dim])
    input_lengths = jnp.full([batch_size], self.max_num_data_points)
    ks = jnp.full([batch_size], self.max_k)
    _, params = self.tfmr.init(key, inputs, input_lengths, ks)
    return params

  def loss(self, params, inputs, input_lengths, true_params, ks, key):
    """Computes the wasserstein loss for this model.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers representing the number of
        data points in each batch element.
      true_params: A three-tuple containing
          true_means: A [batch_size, max_k] tensor containing the true means of
            the cluster components for each batch element.
          true_scales: Unused.
          true_weights: Unused.
      ks: A [batch_size] set of integers representing the true number of
        clusters in each batch element.
      key: A JAX PRNG key.
    Returns:
      The wasserstein distance from the set of predicted mus to the true set
      of mus, a tensor of shape [batch_size].
    """
    true_means, _, _ = true_params
    return self.tfmr.wasserstein_distance_loss(
        params, inputs, input_lengths, true_means, ks, key)

  def predict(self, params, inputs, input_lengths, ks):
    """Predicts the cluster means for the given data sets.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers, the number of data points
        in each batch element.
      ks: A [batch_size] set of integers, the number of clusters in each batch
        element.
    Returns:
      The predicted means, a tensor of shape [batch_size, max_k, data_dim].
    """
    return self.tfmr.call(params, inputs, input_lengths, ks)

  def classify(self, params, inputs, input_lengths, ks):
    """Assigns each point to cluster based on the predicted cluster means.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers, the number of data points
        in each batch element.
      ks: A [batch_size] set of integers, the number of clusters in each batch
        element.
    Returns:
      The predicted clusters, an integer tensor of shape
        [batch_size, max_num_data_points]. Each element is in [0, max_num_k).
    """
    predicted_means = self.predict(params, inputs, input_lengths, ks)
    # [batch_size, max_input_length, max_k]
    dists = util.pair_dists(inputs, predicted_means)
    dists = jnp.where(
        util.make_mask(ks, self.max_k)[:, jnp.newaxis, :], dists,
        jnp.full_like(dists, jnp.inf))
    return jnp.argmin(dists, axis=-1), predicted_means


def flatten_scale(scale):
  dim = scale.shape[-1]
  log_diag = jnp.log(jnp.diag(scale))
  scale = scale.at[jnp.diag_indices(dim)].set(log_diag)
  return scale[jnp.tril_indices(dim)]


def unflatten_scale(flat_scale, original_dim):
  out = jnp.zeros([original_dim, original_dim], dtype=flat_scale.dtype)
  out = out.at[jnp.tril_indices(original_dim)].set(flat_scale)
  exp_diag = jnp.exp(jnp.diag(out))
  return out.at[jnp.diag_indices(original_dim)].set(exp_diag)


def flatten_gmm_params(means, scales, log_weights):
  """
  means: A [max_k, data_dim] tensor containing the means.
  scales: A [max_k, data_dim, data_dim] tensor containing the scale matrices
  log_weights: A [max_k] tensor containing the weight of each component
  """
  flat_scales = vmap(flatten_scale)(scales)
  flat_params = jnp.concatenate(
        [log_weights[:, jnp.newaxis], means, flat_scales], axis=-1)
  return flat_params


def unflatten_gmm_params(flat_params, original_dim):
  """
  flat_params: A [max_k, 1 + data_dim + data_dim*(data_dim-1)/2] set of the flat parameters,
    ordered as log_weights, means, flattened scales.
  original_dim: The original dimension of the data.
  """
  log_weights = flat_params[:, 0]
  mus = flat_params[:, 1:original_dim+1]
  scales = vmap(unflatten_scale, in_axes=(0, None))(
          flat_params[:, original_dim+1:], original_dim)
  return mus, scales, log_weights

def standardize_data(data, num_data_points, max_num_data_points):
  """

  Args:
    data: A [max_num_data_points, data_dim] tensor.
    num_data_points: A scalar denoting the number of data points in data.
    max_num_data_points: A scalar denoting the size of the first dimension of data.
  """
  mask = jnp.arange(max_num_data_points) < num_data_points
  masked_data = data * mask[:, jnp.newaxis]
  mean = jnp.sum(masked_data, 0) / num_data_points
  sq_diff = jnp.square((masked_data - mean[jnp.newaxis, :])) * mask[:,jnp.newaxis]
  std = jnp.sqrt(jnp.sum(sq_diff, axis=0) / num_data_points)
  normalized_data = (data - mean[jnp.newaxis, :]) / std[jnp.newaxis, :]
  masked_norm_data = normalized_data * mask[:, jnp.newaxis]
  return masked_norm_data, mean, std

def unstandardize_data(standardized_data, mean, std, num_data_points, max_num_data_points):
  orig_data = (standardized_data * std) + mean
  mask = jnp.arange(max_num_data_points) < num_data_points
  return orig_data * mask[:,jnp.newaxis]

def unstandardize_params(pred_mean, pred_scale, mean, std):
  unstd_mean = (pred_mean * std) + mean
  unstd_scale = jnp.diag(std) @ pred_scale
  return unstd_mean, unstd_scale


class MeanScaleInferenceMachine(object):

  def __init__(self,
               data_dim=2,
               max_k=2,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               weight_init=jax.nn.initializers.xavier_uniform()):
    """Creates the model.

    Args:
      data_dim: The dimensionality of the data points to be fed in.
      max_k: The maximum number of clusters that could occur in the data.
      max_num_data_points: The maximum number of data points that could be
        fed in at one time.
      num_heads: The number of heads to use in the transformer.
      num_encoders: The number of encoder layers to use in the transformer.
      num_decoders: The number of decoder layers to use in the transformer.
      qkv_dim: The dimensions of the queries, keys, and values in the
        transformer.
      activation_fn: The activation function to use for hidden layers.
      weight_init: The weight initializer.
    """
    self.data_dim = data_dim
    self.max_k = max_k
    self.max_num_data_points = max_num_data_points
    target_dim = data_dim + int((data_dim*(data_dim+1))/2)
    self.tfmr = transformer.EncoderDecoderTransformer.partial(
        target_dim=target_dim,
        max_input_length=max_num_data_points, max_target_length=max_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        activation_fn=activation_fn, weight_init=weight_init)

  def init_params(self, key):
    """Initializes the parameters of the model using dummy data.

    Args:
      key: A JAX PRNG key
    Returns:
      params: The parameters of the model.
    """
    batch_size = 1
    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(
        subkey, [batch_size, self.max_num_data_points, self.data_dim])
    input_lengths = jnp.full([batch_size], self.max_num_data_points)
    ks = jnp.full([batch_size], self.max_k)
    _, params = self.tfmr.init(key, inputs, input_lengths, ks)
    return params

  def loss(self, params, inputs, input_lengths, true_params, ks, key):
    """Computes the wasserstein loss for this model.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers representing the number of
        data points in each batch element.
      true_params: A three-tuple containing
        true_means: A [batch_size, max_k, data_dim] tensor containing the true
          means of the cluster components for each batch element.
        true_scales: A [batch_size, max_k, data_dim, data_dim] tensor containing
          the true scales of the cluster components for each batch element.
          Should be the lower-triangular square root of a PSD matrix.
        true_log_weights: Unused.
      ks: A [batch_size] set of integers representing the true number of
        clusters in each batch element.
      key: A JAX PRNG key.
    Returns:
      The wasserstein distance from the set of predicted mus to the true set
      of mus, a tensor of shape [batch_size].
    """
    true_means, true_scales, _ = true_params
    flat_scales = vmap(vmap(flatten_scale))(true_scales)
    targets = jnp.concatenate([true_means, flat_scales], axis=-1)
    return self.tfmr.wasserstein_distance_loss(
        params, inputs, input_lengths, targets, ks, key)

  def predict(self, params, inputs, input_lengths, ks):
    """Predicts the cluster means for the given data sets.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers, the number of data points
        in each batch element.
      ks: A [batch_size] set of integers, the number of clusters in each batch
        element.
    Returns:
      params: A tuple containing
        The predicted means, a tensor of shape [batch_size, max_k, data_dim].
        The predicted scales, a tensor of shape
          [batch_size, max_k, data_dim, data_dim].
    """
    raw_outs = self.tfmr.call(params, inputs, input_lengths, ks)
    mus = raw_outs[:, :, :self.data_dim]
    us = vmap(vmap(unflatten_scale, in_axes=(0, None)), in_axes=(0, None))
    scales = us(raw_outs[:, :, self.data_dim:], self.data_dim)
    return mus, scales

  def classify(self, params, inputs, input_lengths, ks):
    """Assigns each point to cluster based on the predicted cluster parameters.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers, the number of data points
        in each batch element.
      ks: A [batch_size] set of integers, the number of clusters in each batch
        element.
    Returns:
      clusters: The predicted clusters, an integer tensor of shape
        [batch_size, max_num_data_points]. Each element is in [0, max_num_k).
      params: The predicted cluster parameters (means and covariances).
    """
    means, scales = self.predict(params, inputs, input_lengths, ks)
    covs = jnp.einsum("...ik,...jk->...ij", scales, scales)

    log_ps = vmap(
        vmap(
            vmap(
                jscipy.stats.multivariate_normal.logpdf,
                in_axes=(0, None, None)),
            in_axes=(None, 0, 0)))(inputs, means, covs)
    log_ps = jnp.where(
        util.make_mask(ks, self.max_k)[:, :, jnp.newaxis], log_ps,
        jnp.full_like(log_ps, -jnp.inf))
    clusters = jnp.argmax(log_ps, axis=-2)
    return clusters, (means, covs)


class MeanScaleWeightInferenceMachine(object):

  def __init__(self,
               data_dim=2,
               max_k=2,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               standardize_data=True,
               weight_init=jax.nn.initializers.xavier_uniform()):
    """Creates the model.

    Args:
      data_dim: The dimensionality of the data points to be fed in.
      max_k: The maximum number of clusters that could occur in the data.
      max_num_data_points: The maximum number of data points that could be
        fed in at one time.
      num_heads: The number of heads to use in the transformer.
      num_encoders: The number of encoder layers to use in the transformer.
      num_decoders: The number of decoder layers to use in the transformer.
      qkv_dim: The dimensions of the queries, keys, and values in the
        transformer.
      activation_fn: The activation function to use for hidden layers.
      weight_init: The weight initializer.
    """
    self.max_num_data_points = max_num_data_points
    self.data_dim = data_dim
    self.max_k = max_k
    target_dim = 1 + data_dim + int((data_dim*(data_dim+1))/2)
    self.standardize_data = standardize_data
    self.tfmr = transformer.EncoderDecoderTransformer.partial(
        target_dim=target_dim,
        max_input_length=max_num_data_points, max_target_length=max_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        activation_fn=activation_fn, weight_init=weight_init)

  def init_params(self, key):
    """Initializes the parameters of the model using dummy data.

    Args:
      key: A JAX PRNG key
    Returns:
      params: The parameters of the model.
    """
    key, subkey = jax.random.split(key)
    batch_size = 1
    inputs = jax.random.normal(
        subkey, [batch_size, self.max_num_data_points, self.data_dim])
    input_lengths = jnp.full([batch_size], self.max_num_data_points)
    ks = jnp.full([batch_size], self.max_k)
    _, params = self.tfmr.init(key, inputs, input_lengths, ks)
    return params

  def loss(self, params, inputs, input_lengths, true_params, ks, key):
    """Computes the wasserstein loss for this model.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers representing the number of
        data points in each batch element.
      true_params: A three-tuple containing
        true_means: A [batch_size, max_k, data_dim] tensor containing the true
          means of the cluster components for each batch element.
        true_scales: A [batch_size, max_k, data_dim, data_dim] tensor containing
          the true scales of the cluster components for each batch element.
          Should be the lower-triangular square root of a PSD matrix.
        true_log_weights: A [batch_size, max_k] tensor containing the true
          log weights of the cluster components for each batch element.
      ks: A [batch_size] set of integers representing the true number of
        clusters in each batch element.
      key: A JAX PRNG key.
    Returns:
      The wasserstein distance from the set of predicted mus to the true set
      of mus, a tensor of shape [batch_size].
    """
    true_means, true_scales, true_log_weights = true_params
    flat_true_params = vmap(flatten_gmm_params)(true_means, true_scales, true_log_weights)
    pred_means, pred_scales, pred_log_ws = self.predict(params, inputs, input_lengths, ks)
    flat_preds = vmap(flatten_gmm_params)(pred_means, pred_scales, pred_log_ws)
    return self.tfmr.wasserstein_distance_loss(flat_true_params, ks, flat_preds, key)

  def predict(self, params, inputs, input_lengths, ks):
    """Predicts the cluster means for the given data sets.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers, the number of data points
        in each batch element.
      ks: A [batch_size] set of integers, the number of clusters in each batch
        element.
    Returns:
      params: A tuple containing
        The predicted means, a tensor of shape [batch_size, max_k, data_dim].
        The predicted scales, a tensor of shape
          [batch_size, max_k, data_dim, data_dim].
        The predicted log weights, a tensor of shape [batch_size, max_k].
    """
    if self.standardize_data:
      inputs, data_mean, data_std = vmap(standardize_data, in_axes=(0,0, None))(
          inputs, input_lengths, self.max_num_data_points)

    raw_outs = self.tfmr.call(params, inputs, input_lengths, ks)
    mus, scales, log_weights = vmap(unflatten_gmm_params, in_axes=(0,None))(
        raw_outs, self.data_dim)
    if self.standardize_data:
      mus, scales = vmap(vmap(unstandardize_params, in_axes=(0, 0, None, None)))(
              mus, scales, data_mean, data_std)
    return mus, scales, log_weights

  def classify(self, params, inputs, input_lengths, ks):
    """Assigns each point to cluster based on the predicted cluster parameters.

    Args:
      params: The parameters of the model, returned from init().
      inputs: A [batch_size, max_num_data_points, data_dim] set of input data.
      input_lengths: A [batch_size] set of integers, the number of data points
        in each batch element.
      ks: A [batch_size] set of integers, the number of clusters in each batch
        element.
    Returns:
      clusters: The predicted clusters, an integer tensor of shape
        [batch_size, max_num_data_points]. Each element is in [0, max_num_k).
      params: The predicted cluster parameters (means, covariances, and
        log weights).
    """
    means, scales, log_weights = self.predict(params, inputs, input_lengths, ks)
    covs = jnp.einsum("...ik,...jk->...ij", scales, scales)

    log_ps = vmap(
        vmap(
            vmap(
                jscipy.stats.multivariate_normal.logpdf,
                in_axes=(0, None, None)),
            in_axes=(None, 0, 0)))(inputs, means, covs)
    log_ps = log_ps + log_weights[Ellipsis, jnp.newaxis]
    log_ps = jnp.where(
        util.make_mask(ks, self.max_k)[:, :, jnp.newaxis], log_ps,
        jnp.full_like(log_ps, -jnp.inf))
    clusters = jnp.argmax(log_ps, axis=-2)
    return clusters, (means, covs, log_weights)


def classify_with_defaults(model, params, inputs, batch_size, input_lengths, ks,
                           max_k, default_cov):
  cs, model_params = model.classify(params, inputs, input_lengths, ks)
  if isinstance(model, MeanInferenceMachine):
    mus = model_params
    covs = jnp.tile(default_cov[jnp.newaxis, jnp.newaxis, :, :],
                    [batch_size, max_k, 1, 1])
    log_weights = jnp.zeros([batch_size, max_k])
  elif isinstance(model, MeanScaleInferenceMachine):
    mus, covs = model_params
    log_weights = jnp.zeros([batch_size, max_k])
  elif isinstance(model, MeanScaleWeightInferenceMachine):
    mus, covs, log_weights = model_params
  return cs, (mus, covs, log_weights)
