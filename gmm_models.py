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

import tensorflow_probability as tfp
tfd = tfp.substrates.jax.distributions

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

def flatten_means_scales(means, scales):
  """
  means: A [max_k, data_dim] tensor containing the means.
  scales: A [max_k, data_dim, data_dim] tensor containing the scale matrices
  """
  flat_scales = vmap(flatten_scale)(scales)
  flat_params = jnp.concatenate([means, flat_scales], axis=-1)
  return flat_params

def unflatten_means_scales(flat_params, original_dim):
  return vmap(unflatten_mean_scale, in_axes=(0,None))(flat_params, original_dim)

def unflatten_mean_scale(flat_params, original_dim):
  mu = flat_params[:original_dim]
  scale = unflatten_scale(flat_params[original_dim:], original_dim)
  return mu, scale

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
  return vmap(unflatten_component_params, in_axes=(0,None))(flat_params, original_dim)

def unflatten_component_params(flat_params, original_dim):
  log_weight = flat_params[0]
  mu = flat_params[1:original_dim+1]
  scale = unflatten_scale(flat_params[original_dim+1:], original_dim)
  return mu, scale, log_weight

def kl_param_dist(flat_params_1, flat_params_2, original_dim):
  mu_1, scale_1, log_weight_1 = unflatten_component_params(flat_params_1, original_dim)
  dist_1 = tfd.MultivariateNormalFullCovariance(loc=mu_1, covariance_matrix=scale_1 @ scale_1.T)
  mu_2, scale_2, log_weight_2 = unflatten_component_params(flat_params_2, original_dim)
  dist_2 = tfd.MultivariateNormalFullCovariance(loc=mu_2, covariance_matrix=scale_2 @ scale_2.T)
  kl = dist_1.kl_divergence(dist_2)
  return kl + (log_weight_1 - log_weight_2)**2

def symm_kl_param_dist(flat_params_1, flat_params_2, original_dim):
  mu_1, scale_1, log_weight_1 = unflatten_component_params(flat_params_1, original_dim)
  dist_1 = tfd.MultivariateNormalFullCovariance(loc=mu_1, covariance_matrix=scale_1 @ scale_1.T)
  mu_2, scale_2, log_weight_2 = unflatten_component_params(flat_params_2, original_dim)
  dist_2 = tfd.MultivariateNormalFullCovariance(loc=mu_2, covariance_matrix=scale_2 @ scale_2.T)
  kl_1 = jnp.minimum(dist_1.kl_divergence(dist_2), 1e4)
  kl_2 = jnp.minimum(dist_2.kl_divergence(dist_1), 1e4)
  return 0.5*(kl_1 + kl_2) + (log_weight_1 - log_weight_2)**2

l2_dist = lambda x,y: jnp.linalg.norm(x-y)

NAMED_DIST_FNS = {
    "l2": l2_dist,
    "kl": kl_param_dist,
    "symm_kl": symm_kl_param_dist
}

class GMMInferenceMachine(object):

  def __init__(self,
               model,
               data_dim=2,
               max_k=2,
               max_num_data_points=25,
               dist="symm_kl",
               entropy_alpha=0.01):
    """Creates the model.

    Args:
      model: The model which accepts data and produces parameters.
      data_dim: The dimensionality of the data points to be fed in.
      max_k: The maximum number of clusters that could occur in the data.
      max_num_data_points: The maximum number of data points that could be
        fed in at one time.
      dist: The name of the distance function to use in computing the loss, selected from the 
        keys of NAMED_DIST_FNS.
      entropy_alpha: The entropy regularization weight used in computing the loss.
    """
    self.model = model
    self.max_num_data_points = max_num_data_points
    self.data_dim = data_dim
    self.max_k = max_k
    if dist == "symm_kl" or dist == "kl":
      self.dist = lambda x,y: NAMED_DIST_FNS[dist](x, y, data_dim)
    else:
      self.dist = NAMED_DIST_FNS[dist]
    self.entropy_alpha = entropy_alpha


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
    _, params = self.model.init(key, inputs, input_lengths, ks)
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
    raise NotImplementedError("Loss not implemented yet.") 

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
    raise NotImplementedError("Classify not implemented yet.") 

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
    raise NotImplementedError("Predict not implemented yet.") 


class MeanInferenceMachine(GMMInferenceMachine):
  """Model which predicts cluster means from a batch of data."""

  def __init__(self,
               model,
               data_dim=2,
               max_k=2,
               max_num_data_points=25,
               dist=None,
               entropy_alpha=0.01):
    """Creates the model.

    Args:
      model: The model which accepts data and produces parameters.
      data_dim: The dimensionality of the data points to be fed in.
      max_k: The maximum number of clusters that could occur in the data.
      max_num_data_points: The maximum number of data points that could be
        fed in at one time.
      dist: Ignored, must be l2_dist for this model.
      entropy_alpha: The entropy regularization weight used in computing the loss.
    """
    super().__init__(model, data_dim=data_dim, max_k=max_k, 
        max_num_data_points=max_num_data_points, dist="l2",
        entropy_alpha=entropy_alpha)

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
    batch_size = inputs.shape[0]
    true_means, _, _ = true_params
    preds = self.model.call(params, inputs, input_lengths, ks)
    log_weights = jnp.zeros([self.max_k])
    
    def loss_fn(preds, targets, length, key):
      return util.simple_masked_sinkhorn_with_dist(preds, log_weights, targets, log_weights,
          self.dist, length, self.max_k, key, alpha=self.entropy_alpha)[0]

    return vmap(loss_fn)(preds, true_means, ks, jax.random.split(key, num=batch_size))

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
    return self.model.call(params, inputs, input_lengths, ks)

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


class OriginalMeanInferenceMachine(MeanInferenceMachine):
  """Model which predicts cluster means from a batch of data."""

  def __init__(self,
               data_dim=2,
               max_k=2,
               algo_k=None,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               normalization="no_norm",
               dist=None,
               weight_init=jax.nn.initializers.xavier_uniform(),
               entropy_alpha=0.01):
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
      normalization: The type of normalization to use, either layernorm or no_norm.
      dist: Ignored, must be l2_dist for this model.
      weight_init: The weight initializer.
      entropy_alpha: The weight of the entropy regularization in the loss.
    """
    model = transformer.EncoderDecoderTransformer.partial(
        target_dim=data_dim,
        max_input_length=max_num_data_points, max_target_length=max_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        normalization=normalization,
        activation_fn=activation_fn, weight_init=weight_init)
    super().__init__(
               model,
               data_dim=data_dim,
               max_k=max_k,
               max_num_data_points=max_num_data_points,
               dist="l2_dist",
               entropy_alpha=entropy_alpha)

class UnconditionalMeanInferenceMachine(MeanInferenceMachine):
  """Model which predicts cluster means from a batch of data."""

  def __init__(self,
               data_dim=2,
               max_k=2,
               algo_k=None,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               normalization="no_norm",
               dist=None,
               weight_init=jax.nn.initializers.xavier_uniform(),
               entropy_alpha=0.01):
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
      normalization: The type of normalization to use, either layernorm or no_norm.
      dist: Ignored, must be l2_dist for this model.
      weight_init: The weight initializer.
      entropy_alpha: The weight of the entropy regularization in the loss.
    """
    model = transformer.UnconditionalEncoderDecoderTransformer.partial(
        target_dim=data_dim,
        max_input_length=max_num_data_points, max_target_length=max_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        normalization=normalization,
        activation_fn=activation_fn, weight_init=weight_init)
    super().__init__(
               model,
               data_dim=data_dim,
               max_k=max_k,
               max_num_data_points=max_num_data_points,
               dist="l2",
               entropy_alpha=entropy_alpha)

class MeanScaleWeightInferenceMachine(GMMInferenceMachine):

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
    batch_size = inputs.shape[0]
    true_means, true_scales, true_log_weights = true_params
    flat_true_params = vmap(flatten_gmm_params)(true_means, true_scales, true_log_weights)
    preds = self.model.call(params, inputs, input_lengths, ks)

    log_weights = jnp.zeros([self.max_k])

    def loss_fn(preds, targets, length, key):
      return util.simple_masked_sinkhorn_with_dist(preds, log_weights, targets, log_weights,
          self.dist, length, self.max_k, key, alpha=self.entropy_alpha)[0]

    return vmap(loss_fn)(preds, flat_true_params, ks, jax.random.split(key, num=batch_size))

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
    raw_outs = self.model.call(params, inputs, input_lengths, ks)
    mus, scales, log_weights = vmap(unflatten_gmm_params, in_axes=(0,None))(
        raw_outs, self.data_dim)
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


class MSWOriginal(MeanScaleWeightInferenceMachine):
  """The original MSW Inference machine that feeds back predictions during decoding."""

  def __init__(self,
               data_dim=2,
               max_k=2,
               algo_k=None,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               normalization="no_norm",
               dist="l2",
               weight_init=jax.nn.initializers.xavier_uniform(),
               entropy_alpha=0.01):
    """The original MSW Inference machine which feeds back predictions while decoding.

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
      normalization: The type of normalization to use, either layernorm or no_norm.
      dist: The distance function to use
      weight_init: The weight initializer.
      entropy_alpha: The weight of the entropy regularization in the loss.
    """
    target_dim = 1 + data_dim + int((data_dim*(data_dim+1))/2)
    model = transformer.EncoderDecoderTransformer.partial(
        target_dim=target_dim,
        max_input_length=max_num_data_points, max_target_length=max_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        activation_fn=activation_fn, normalization=normalization, weight_init=weight_init)
    super().__init__(
        model, data_dim=data_dim, max_k=max_k, 
        max_num_data_points=max_num_data_points,
        dist=dist, entropy_alpha=entropy_alpha)


class MSWUnconditional(MeanScaleWeightInferenceMachine):
  """An MSW Inference machine that doesn't feed back predictions."""

  def __init__(self,
               data_dim=2,
               max_k=2,
               algo_k=None,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               normalization="no_norm",
               dist="l2",
               weight_init=jax.nn.initializers.xavier_uniform(),
               entropy_alpha=0.01):
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
      normalization: The type of normalization to use, either layernorm or no_norm.
      dist: The distance function to use
      weight_init: The weight initializer.
      entropy_alpha: The weight of the entropy regularization in the loss.
    """
    target_dim = 1 + data_dim + int((data_dim*(data_dim+1))/2)
    model = transformer.UnconditionalEncoderDecoderTransformer.partial(
        target_dim=target_dim,
        max_input_length=max_num_data_points, max_target_length=max_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        activation_fn=activation_fn, normalization=normalization, weight_init=weight_init)
    super().__init__(
        model, data_dim=data_dim, max_k=max_k, 
        max_num_data_points=max_num_data_points,
        dist=dist, entropy_alpha=entropy_alpha)


def classify_with_defaults(model, params, inputs, batch_size, input_lengths, ks,
                           max_k, default_cov):
  cs, model_params = model.classify(params, inputs, input_lengths, ks)
  if isinstance(model, MeanInferenceMachine):
    mus = model_params
    covs = jnp.tile(default_cov[jnp.newaxis, jnp.newaxis, :, :],
                    [batch_size, max_k, 1, 1])
    log_weights = jnp.zeros([batch_size, max_k])
  #elif isinstance(model, MeanScaleInferenceMachine):
  #  mus, covs = model_params
  #  log_weights = jnp.zeros([batch_size, max_k])
  elif (isinstance(model, MeanScaleWeightInferenceMachine) or
        isinstance(model, NoDecoderInferenceMachine)):
    mus, covs, log_weights = model_params
  return cs, (mus, covs, log_weights)


class FixedKInferenceMachine(MeanScaleWeightInferenceMachine):

  def __init__(self,
               model,
               data_dim=2,
               max_k=2,
               algo_k=2,
               max_num_data_points=25,
               dist="symm_kl",
               entropy_alpha=0.01):
    self.algo_k = algo_k
    super().__init__(model, data_dim=data_dim, max_k=max_k,
        max_num_data_points=max_num_data_points, dist=dist, entropy_alpha=entropy_alpha)


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
    batch_size = inputs.shape[0]
    true_means, true_scales, true_log_weights = true_params
    true_flat_params = vmap(flatten_means_scales)(true_means, true_scales)
    # ks (target lengths) is not used here.
    preds = self.model.call(params, inputs, input_lengths, ks)
    pred_log_weights = preds[:, :, 0]
    pred_flat_params = preds[:, :, 1:]

    def loss_fn(preds, pred_log_weights, targets, target_log_weights, length, key):
      return util.masked_sinkhorn_with_dist(
          preds, pred_log_weights, self.algo_k, self.algo_k,
          targets, target_log_weights, length, self.max_k,
          self.dist, key, alpha=self.entropy_alpha)[0]

    return vmap(loss_fn)(pred_flat_params, pred_log_weights,
                         true_flat_params, true_log_weights, 
                         ks, jax.random.split(key, num=batch_size))


class UnconditionalFixedK(FixedKInferenceMachine):
  """An unconditional MSW Inference machine that doesn't feed back predictions."""

  def __init__(self,
               data_dim=2,
               max_k=2,
               algo_k=2,
               max_num_data_points=25,
               num_heads=8,
               num_encoders=6,
               num_decoders=6,
               qkv_dim=512,
               activation_fn=flax.nn.relu,
               normalization="no_norm",
               dist="l2",
               weight_init=jax.nn.initializers.xavier_uniform(),
               entropy_alpha=0.01):
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
      normalization: The type of normalization to use, either layernorm or no_norm.
      dist: The distance function to use
      weight_init: The weight initializer.
      entropy_alpha: The weight of the entropy regularization in the loss.
    """
    target_dim = 1 + data_dim + int((data_dim*(data_dim+1))/2)
    model = transformer.UnconditionalEncoderDecoderTransformer.partial(
        target_dim=target_dim,
        max_input_length=max_num_data_points, max_target_length=algo_k,
        num_heads=num_heads, num_encoders=num_encoders,
        num_decoders=num_decoders, qkv_dim=qkv_dim,
        activation_fn=activation_fn, normalization=normalization, weight_init=weight_init)
    super().__init__(
        model, data_dim=data_dim, max_k=max_k, algo_k=algo_k,
        max_num_data_points=max_num_data_points,
        dist=dist, entropy_alpha=entropy_alpha)

