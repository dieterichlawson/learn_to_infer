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

"""Flax implementation of the transformer encoder.
"""
from functools import partial

import util

from collections.abc import Iterable  # pylint: disable=g-importing-member

import warnings


import numpy as onp
import flax
from flax import nn
import jax
import jax.numpy as jnp
import jax.random
from jax import vmap
import jax.experimental
import jax.experimental.host_callback as hcb

from flax.nn.activation import softmax
from flax.nn.base import Collection, Module, collection_from_iterable, iterate_collection
from flax.nn.initializers import zeros
from flax.nn.stochastic import make_rng
from flax.nn.linear import DenseGeneral, default_kernel_init
from flax import struct


def normalize(inputs, normalization_type, name=None):
  if normalization_type == "no_norm":
    return inputs
  elif normalization_type == 'layer_norm':
    return nn.LayerNorm(inputs, bias=True, scale=False, name=name)
  elif normalization_type == 'batch_norm':
    return inputs

class TransformerEncoderLayer(nn.Module):

  def apply(self,
            inputs,
            mask,
            activation_fn=flax.nn.relu,
            num_heads=8,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies one transformer encoder layer.

    Args:
      inputs: The inputs to the transformer, a
        [batch_size, max_num_data_points, data_dim] tensor.
      mask: The mask for the inputs indicating which elements are padding. A
        tensor of shape [batch_size, max_num_data_points].
      activation_fn: The activation function to use, defaults to relu.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_num_data_points, data_dim] tensor of outputs.
    """
    value_dim = inputs.shape[-1]

    attn_outs = flax.nn.SelfAttention(
        inputs_q=inputs,
        num_heads=num_heads,
        qkv_features=value_dim,
        padding_mask=mask,
        kernel_init=weight_init)

    attn_outs = inputs + attn_outs
    
    attn_outs = normalize(attn_outs, normalization)

    out1 = activation_fn(flax.nn.Dense(attn_outs,
                                       features=value_dim,
                                       kernel_init=weight_init))
    out2 = flax.nn.Dense(out1,
                         features=value_dim,
                         kernel_init=weight_init)

    outs = attn_outs + out2
    return normalize(outs, normalization)


class RepeatedTransformerEncoderStack(nn.Module):

  def apply(self,
            inputs,
            mask,
            num_encoders=6,
            num_heads=8,
            value_dim=128,
            activation_fn=flax.nn.relu,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies a stack of transformer encoder layers.

    Args:
      inputs: The inputs to the transformer, a
        [batch_size, max_num_data_points, data_dim] tensor.
      mask: The mask for the inputs indicating which elements are padding. A
        tensor of shape [batch_size, max_num_data_points].
      num_encoders: The number of encoder layers in the stack.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      value_dim: The dimension of the transformer's keys, values, and queries.
      activation_fn: The activation function to use, defaults to relu.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_num_data_points, value_dim] tensor of outputs.
    """
    inputs = flax.nn.Dense(inputs, features=value_dim, kernel_init=weight_init)
    encoder = TransformerEncoderLayer.shared(
        activation_fn=activation_fn,
        num_heads=num_heads,
        normalization=normalization,
        weight_init=weight_init)
    for _ in range(num_encoders):
      inputs = encoder(inputs, mask)
    return inputs


class TransformerEncoderStack(nn.Module):

  def apply(self,
            inputs,
            mask,
            num_encoders=6,
            num_heads=8,
            value_dim=128,
            activation_fn=flax.nn.relu,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies a stack of transformer encoder layers.

    Args:
      inputs: The inputs to the transformer, a
        [batch_size, max_num_data_points, data_dim] tensor.
      mask: The mask for the inputs indicating which elements are padding. A
        tensor of shape [batch_size, max_num_data_points].
      num_encoders: The number of encoder layers in the stack.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      value_dim: The dimension of the transformer's keys, values, and queries.
      activation_fn: The activation function to use, defaults to relu.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_num_data_points, value_dim] tensor of outputs.
    """
    inputs = flax.nn.Dense(inputs, features=value_dim, kernel_init=weight_init)
    for _ in range(num_encoders):
      inputs = TransformerEncoderLayer(inputs,
                                       mask,
                                       activation_fn=activation_fn,
                                       num_heads=num_heads,
                                       normalization=normalization,
                                       weight_init=weight_init)
    return inputs


class TransformerDecoderLayer(nn.Module):

  def apply(self,
            target_inputs,
            target_mask,
            encoder_inputs,
            encoder_mask,
            activation_fn=flax.nn.relu,
            num_heads=8,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies one transformer decoder layer.

    Args:
      target_inputs: The inputs derived from the transformer outputs, a
        [batch_size, max_k, value_dim] tensor.
      target_mask: The mask for the targets indicating which elements are
        padding. A tensor of shape [batch_size, max_k].
      encoder_inputs: The inputs derived from the transformer inputs, a
        [batch_size, max_num_data_points, value_dim] tensor.
      encoder_mask: The mask for the inputs indicating which elements are
        padding. A tensor of shape [batch_size, num_data_points].
      activation_fn: The activation function to use, defaults to relu.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_k, value_dim] tensor of outputs.
    """
    value_dim = target_inputs.shape[-1]
    target_inputs_attn = flax.nn.SelfAttention(
        inputs_q=target_inputs,
        num_heads=num_heads,
        causal_mask=True,
        padding_mask=target_mask,
        qkv_features=value_dim,
        kernel_init=weight_init)

    target_inputs_out = normalize(target_inputs_attn + target_inputs, normalization)

    enc_dec_attn_out = flax.nn.MultiHeadDotProductAttention(
        inputs_q=target_inputs_out,
        inputs_kv=encoder_inputs,
        padding_mask=target_mask,
        key_padding_mask=encoder_mask,
        num_heads=num_heads,
        qkv_features=value_dim,
        kernel_init=weight_init)

    enc_dec_attn_out = normalize(target_inputs_out + enc_dec_attn_out, normalization)

    out_layer1 = activation_fn(flax.nn.Dense(enc_dec_attn_out,
                                             features=value_dim,
                                             kernel_init=weight_init))
    out_layer2 = flax.nn.Dense(out_layer1,
                               features=value_dim,
                               kernel_init=weight_init)

    return normalize(out_layer2 + enc_dec_attn_out, normalization)


class TransformerDecoderStack(nn.Module):

  def apply(self,
            target_inputs,
            target_mask,
            encoder_inputs,
            encoder_mask,
            activation_fn=flax.nn.relu,
            num_decoders=6,
            num_heads=8,
            value_dim=128,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies a stack of transformer decoder layers.

    Args:
      target_inputs: The inputs derived from the transformer outputs, a
        [batch_size, max_k, data_dim] tensor.
      target_mask: The mask for the targets indicating which elements are
        padding. A tensor of shape [batch_size, max_k].
      encoder_inputs: The inputs derived from the transformer inputs, a
        [batch_size, max_num_data_points, value_dim] tensor.
      encoder_mask: The mask for the inputs indicating which elements are
        padding. A tensor of shape [batch_size, num_data_points].
      activation_fn: The activation function to use, defaults to relu.
      num_decoders: The number of decoders in the stack.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      value_dim: The dimension of the transformer's keys, values, and queries.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_k, value_dim] tensor of outputs.
    """
    inputs = flax.nn.Dense(
        target_inputs, features=value_dim, kernel_init=weight_init)

    for _ in range(num_decoders):
      inputs = TransformerDecoderLayer(inputs,
                                       target_mask,
                                       encoder_inputs,
                                       encoder_mask,
                                       activation_fn=activation_fn,
                                       num_heads=num_heads,
                                       normalization=normalization,
                                       weight_init=weight_init)
    return inputs


class UnconditionalEncoderDecoderTransformer(nn.Module):

  def apply(self,
            inputs,
            input_lengths,
            target_lengths,
            targets=None,
            target_dim=32,
            max_target_length=100,
            num_heads=8,
            num_encoders=6,
            num_decoders=6,
            qkv_dim=512,
            activation_fn=flax.nn.relu,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_uniform(),
            tie_layer_weights=False):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data, [batch_size, max_num_data_points, data_dim].
      input_lengths: A [batch_size] vector containing the number of samples
        in each batch element.
      target_lengths: Unused.
      targets: Unused.
      target_dim: The length of each output vector.
      max_target_length: An int at least as large as the largest element of
        target_lengths, used for determining output shapes.
      num_heads: The number of heads for the self attention.
      num_encoders: The number of transformer encoder layers.
      num_decoders: The number of transformer decoder layers.
      qkv_dim: The dimension of the query/key/value.
      activation_fn: The activation function to use, defaults to relu.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: The transformer output, a tensor of shape
        [batch_size, max_target_length, target_dim].
    """
    batch_size = inputs.shape[0]
    max_input_length = inputs.shape[1]
    input_mask = util.make_mask(input_lengths, max_input_length)
    
    if tie_layer_weights:
      encoder_hs = RepeatedTransformerEncoderStack(
          inputs,
          input_mask,
          num_encoders=num_encoders,
          num_heads=num_heads,
          value_dim=qkv_dim,
          normalization=normalization,
          weight_init=weight_init)
    else:
      encoder_hs = TransformerEncoderStack(
          inputs,
          input_mask,
          num_encoders=num_encoders,
          num_heads=num_heads,
          value_dim=qkv_dim,
          normalization=normalization,
          weight_init=weight_init)

    # average over data dimension, resulting in [batch_size, data_dim]
    encoder_out = jnp.mean(encoder_hs, axis=1)

    decoder_out = flax.nn.Dense(encoder_out, features=max_target_length*64, 
        kernel_init=weight_init)

    for i in range(num_decoders):
      layer_out = activation_fn(flax.nn.Dense(decoder_out,
                                       features=max_target_length*64,
                                       kernel_init=weight_init))
      layer_out = normalize(layer_out, normalization)

      layer_out = flax.nn.Dense(layer_out,
                                features=max_target_length*64,
                                kernel_init=weight_init)
      layer_out = activation_fn(layer_out)
      decoder_out = normalize(layer_out + decoder_out, normalization)


    # dense layer to arrive at [batch_size, target_length, target_dim]
    out = flax.nn.Dense(
        decoder_out,
        features=target_dim*max_target_length,
        kernel_init=weight_init)
    out = jnp.reshape(out, [batch_size, max_target_length, target_dim])

    return out


class EncoderDecoderTransformer(nn.Module):

  def apply(self,
            inputs,
            input_lengths,
            target_lengths,
            target_dim=32,
            max_target_length=100,
            num_heads=8,
            num_encoders=6,
            num_decoders=6,
            qkv_dim=512,
            activation_fn=flax.nn.relu,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_uniform()):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data, [batch_size, max_num_data_points, data_dim].
      input_lengths: A [batch_size] vector containing the number of samples
        in each batch element.
      target_lengths: A [batch_size] vector containing the length of each
        target sequence.
      targets: The outputs to be produced by the transformer. Supplied only
        during training. If None, then the transformer's own outputs are fed
        back in.
      target_dim: The length of each output vector.
      max_target_length: An int at least as large as the largest element of
        target_lengths, used for determining output shapes.
      num_heads: The number of heads for the self attention.
      num_encoders: The number of transformer encoder layers.
      num_decoders: The number of transformer decoder layers.
      qkv_dim: The dimension of the query/key/value.
      activation_fn: The activation function to use, defaults to relu.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: The transformer output, a tensor of shape
        [batch_size, max_target_length, target_dim].
    """
    max_input_length = inputs.shape[1]
    input_mask = util.make_mask(input_lengths, max_input_length)
    target_mask = util.make_mask(target_lengths, max_target_length)

    encoder_hs = TransformerEncoderStack(inputs,
                                         input_mask,
                                         num_encoders=num_encoders,
                                         num_heads=num_heads,
                                         value_dim=qkv_dim,
                                         normalization=normalization,
                                         weight_init=weight_init)
    batch_size = inputs.shape[0]

    target_inputs = jnp.zeros([batch_size, max_target_length, target_dim])
    target_inputs = target_inputs.at[:, 0, 0].set(target_lengths)

    decoder_stack = TransformerDecoderStack.shared(
        activation_fn=flax.nn.relu, num_decoders=num_decoders,
        num_heads=num_heads, value_dim=qkv_dim,
        normalization=normalization, weight_init=weight_init)

    dense_1 = flax.nn.Dense.shared(features=qkv_dim, kernel_init=weight_init)
    dense_2 = flax.nn.Dense.shared(features=target_dim, kernel_init=weight_init)

    def decode_body(target_inputs, i):
      # decoder_out is [batch_size, max_target_length, value_dim]
      decoder_out = decoder_stack(target_inputs, target_mask, encoder_hs, input_mask)

      # out is [batch_size, qkv_dim]
      out = activation_fn(dense_1(decoder_out[:, i]))
      # dense layer to arrive at [batch_size, target_dim]
      out = dense_2(out)

      target_inputs = target_inputs.at[:, i + 1].set(out)
      return target_inputs, out

    if self.is_initializing():
      decode_body(target_inputs, 0)

    _, outs = jax.lax.scan(
        decode_body,
        jnp.zeros([batch_size, max_target_length, target_dim]),
        jnp.arange(max_target_length),
    )
    # outs is currently [max_target_length, batch_size, target_dim],
    # transpose to put the batch dimension first.
    return jnp.transpose(outs, axes=(1, 0, 2))


class ResNet(nn.Module):

  def apply(self,
            inputs,
            out_dim=2,
            activation=jax.nn.relu,
            hidden_dim=128,
            num_blocks=2,
            weight_init=jax.nn.initializers.xavier_normal(),
            name="fc_net"):

    x = flax.nn.Dense(inputs, hidden_dim, kernel_init=weight_init)

    for _ in range(num_blocks):
      out = flax.nn.Dense(x, hidden_dim, kernel_init=weight_init)
      out = activation(out)
      out = flax.nn.Dense(out, hidden_dim, kernel_init=weight_init)
      x = activation(out + x)

    out = flax.nn.Dense(x, out_dim)
    return out



def _invert_perm(perm):
  perm_inv = [0] * len(perm)
  for i, j in enumerate(perm):
    perm_inv[j] = i
  return tuple(perm_inv)

def dot_product_attention(query,
                          key,
                          value,
                          dtype=jnp.float32,
                          bias=None,
                          axis=None,
                          deterministic=False,
                          precision=None):
  """DEPRECATION WARNING:
 "The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Computes dot-product attention given query, key, and value.

  This is the core function for applying attention based on
  https://arxiv.org/abs/1706.03762. It calculates the attention weights given
  query and key and combines the values using the attention weights. This
  function supports multi-dimensional inputs.


  Args:
    query: queries for calculating attention with shape of `[batch_size, dim1,
      dim2, ..., dimN, num_heads, mem_channels]`.
    key: keys for calculating attention with shape of `[batch_size, dim1, dim2,
      ..., dimN, num_heads, mem_channels]`.
    value: values to be used in attention with shape of `[batch_size, dim1,
      dim2,..., dimN, num_heads, value_channels]`.
    dtype: the dtype of the computation (default: float32)
    bias: bias for the attention weights. This can be used for incorporating
      autoregressive mask, padding mask, proximity bias.
    axis: axises over which the attention is applied.
    broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    dropout_rng: JAX PRNGKey: to be used for dropout
    dropout_rate: dropout rate
    deterministic: bool, deterministic or not (to apply dropout)
    precision: numerical precision of the computation see `jax.lax.Precision`
      for details.

  Returns:
    Output of shape `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]`.
  """
  assert key.shape[:-1] == value.shape[:-1]
  assert (query.shape[0:1] == key.shape[0:1] and
          query.shape[-1] == key.shape[-1])

  if axis is None:
    axis = tuple(range(1, key.ndim - 2))
  if not isinstance(axis, Iterable):
    axis = (axis,)
  assert key.ndim == query.ndim
  assert key.ndim == value.ndim
  for ax in axis:
    if not (query.ndim >= 3 and 1 <= ax < query.ndim - 2):
      raise ValueError('Attention axis must be between the batch '
                       'axis and the last-two axes.')
  depth = query.shape[-1]
  n = key.ndim
  # batch_dims is  <bs, <non-attention dims>, num_heads>
  batch_dims = tuple(onp.delete(range(n), axis + (n - 1,)))
  # q & k -> (bs, <non-attention dims>, num_heads, <attention dims>, channels)
  qk_perm = batch_dims + axis + (n - 1,)
  key = key.transpose(qk_perm)
  query = query.transpose(qk_perm)
  # v -> (bs, <non-attention dims>, num_heads, channels, <attention dims>)
  v_perm = batch_dims + (n - 1,) + axis
  value = value.transpose(v_perm)

  query = query / jnp.sqrt(depth).astype(dtype)
  batch_dims_t = tuple(range(len(batch_dims)))
  attn_weights = jax.lax.dot_general(
      query,
      key, (((n - 1,), (n - 1,)), (batch_dims_t, batch_dims_t)),
      precision=precision)

  # apply attention bias: masking, droput, proximity bias, ect.
  if bias is not None:
    attn_weights = attn_weights + bias

  # normalize the attention weights
  norm_dims = tuple(range(attn_weights.ndim - len(axis), attn_weights.ndim))
  attn_weights = softmax(attn_weights, axis=norm_dims)
  attn_weights = attn_weights.astype(dtype)

  # compute the new values given the attention weights
  wv_contracting_dims = (norm_dims, range(value.ndim - len(axis), value.ndim))
  y = jax.lax.dot_general(
      attn_weights,
      value, (wv_contracting_dims, (batch_dims_t, batch_dims_t)),
      precision=precision)

  # back to (bs, dim1, dim2, ..., dimN, num_heads, channels)
  perm_inv = _invert_perm(qk_perm)
  y = y.transpose(perm_inv)
  return y, attn_weights

class MultiHeadDotProductAttention(nn.Module):
  """The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Multi-head dot-product attention."""

  def apply(self,
            inputs_q,
            inputs_kv,
            num_heads,
            dtype=jnp.float32,
            qkv_features=None,
            out_features=None,
            attention_axis=None,
            causal_mask=False,
            padding_mask=None,
            key_padding_mask=None,
            segmentation=None,
            key_segmentation=None,
            deterministic=False,
            precision=None,
            kernel_init=default_kernel_init,
            bias_init=zeros,
            bias=True,
            attention_fn=dot_product_attention):
    """Applies multi-head dot product attention on the input data.

    Projects the inputs into multi-headed query, key, and value vectors,
    applies dot-product attention and project the results to an output vector.

    This can be used for encoder-decoder attention by specifying both `inputs_q`
    and `inputs_kv` orfor self-attention by only specifying `inputs_q` and
    setting `inputs_kv` to None.

    Args:
      inputs_q: input queries of shape `[bs, dim1, dim2, ..., dimN, features]`.
      inputs_kv: key/values of shape `[bs, dim1, dim2, ..., dimN, features]`
        or None for self-attention, inn which case key/values will be derived
        from inputs_q.
      num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
        should be divisible by the number of heads.
      dtype: the dtype of the computation (default: float32)
      qkv_features: dimension of the key, query, and value.
      out_features: dimension of the last projection
      attention_axis: axes over which the attention is applied ( 'None' means
        attention over all axes, but batch, heads, and features).
      causal_mask: boolean specifying whether to apply a causal mask on the
        attention weights. If True, the output at timestep `t` will not depend
        on inputs at timesteps strictly greater than `t`.
      padding_mask: boolean specifying query tokens that are pad token w/ False.
      key_padding_mask: boolean specifying key-value tokens that are pad token
        w/ False.
      segmentation: segment indices for packed inputs_q data.
      key_segmentation: segment indices for packed inputs_kv data.
      cache: an instance of `flax.nn.attention.Cache` used for efficient
        autoregressive decoding.
      broadcast_dropout: bool: use a broadcasted dropout along batch dims.
      dropout_rng: JAX PRNGKey: to be used for dropout
      dropout_rate: dropout rate
      deterministic: bool, deterministic or not (to apply dropout)
      precision: numerical precision of the computation see `jax.lax.Precision`
        for details.
      kernel_init: initializer for the kernel of the Dense layers.
      bias_init: initializer for the bias of the Dense layers.
      bias: bool: whether pointwise QKVO dense transforms use bias.
      attention_fn: dot_product_attention or compatible function. Accepts
      query, key, value, and returns output of shape
      `[bs, dim1, dim2, ..., dimN,, num_heads, value_channels]``

    Returns:
      output of shape `[bs, dim1, dim2, ..., dimN, features]`.
    """

    if inputs_kv is None:
      inputs_kv = inputs_q

    is_self_attention = inputs_kv is inputs_q

    if attention_axis is None:
      attention_axis = tuple(range(1, inputs_q.ndim - 1))

    features = out_features or inputs_q.shape[-1]
    qkv_features = qkv_features or inputs_q.shape[-1]

    assert qkv_features % num_heads == 0, (
        'Memory dimension must be divisible by number of heads.')
    head_dim = qkv_features // num_heads

    dense = DenseGeneral.partial(
        axis=-1,
        features=(num_heads, head_dim),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        precision=precision)
    # project inputs_q to multi-headed q/k/v
    # dimensions are then [bs, dims..., n_heads, n_features_per_head]
    query, key, value = (dense(inputs_q, dtype=dtype, name='query'),
                         dense(inputs_kv, dtype=dtype, name='key'),
                         dense(inputs_kv, dtype=dtype, name='value'))

    # create attention masks
    mask_components = []

    if causal_mask:
      mask_components.append(_make_causal_mask(key, attention_axis))

    if (padding_mask is not None or key_padding_mask is not None):
      if key_padding_mask is None:
        if is_self_attention:
          key_padding_mask = padding_mask
        else:
          key_padding_shape = [inputs_kv.shape[dim] for dim in attention_axis]
          key_padding_mask = jnp.full(key_padding_shape, True)
      if padding_mask is None:
        if is_self_attention:
          padding_mask = key_padding_mask
        else:
          padding_shape = [inputs_q.shape[dim] for dim in attention_axis]
          padding_mask = jnp.full(padding_shape, True)

      padding_mask = make_padding_mask(
          padding_mask_query=padding_mask,
          padding_mask_key=key_padding_mask,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis)
      mask_components.append(padding_mask)

    if segmentation is not None:
      if key_segmentation is None:
        assert is_self_attention
        key_segmentation = segmentation
      segmentation_mask = make_padding_mask(
          padding_mask_query=segmentation,
          padding_mask_key=key_segmentation,
          query_shape=query.shape,
          key_shape=key.shape,
          attention_axis=attention_axis,
          segmentation_mask=True)
      mask_components.append(segmentation_mask)

    if mask_components:
      attention_mask = mask_components[0]
      for component in mask_components[1:]:
        attention_mask = jnp.logical_and(attention_mask, component)

      # attention mask in the form of attention bias
      attention_bias = jax.lax.select(
          attention_mask > 0, jnp.full(attention_mask.shape, 0.).astype(dtype),
          jnp.full(attention_mask.shape, -1e10).astype(dtype))
    else:
      attention_bias = None

    # apply attention
    x, attn_weights = attention_fn(
        query,
        key,
        value,
        dtype=dtype,
        axis=attention_axis,
        bias=attention_bias,
        precision=precision,
        deterministic=deterministic)

    # back to the original inputs dimensions
    out = DenseGeneral(
        x,
        features=features,
        axis=(-2, -1),
        kernel_init=kernel_init,
        bias_init=bias_init,
        bias=bias,
        dtype=dtype,
        precision=precision,
        name='out')

    return out, query, key, attn_weights


# TODO(flax-dev): Consider refactoring MultiHeadDotProductAttention and moving
# causal_mask and cache support into this class instead.
SelfAttention = MultiHeadDotProductAttention.partial(inputs_kv=None)

def make_padding_mask(padding_mask_query,
                      padding_mask_key,
                      query_shape,
                      key_shape,
                      attention_axis=None,
                      segmentation_mask=False):
  """The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Makes padding mask for attention weights.

  In case of 1d inputs (i.e., `[bs, len, features]`, the attention weights will
  be `[bs, len, len]` and this function makes a square matrix [len, len].

  Args:
    padding_mask_query: padding mask of query <bs, qdim1,.., qdimn>
    padding_mask_key: padding mask of query <bs, key1,.., keyn>
    query_shape: shape of the query
    key_shape: shape of the key, which is equal to the shape of value.
    attention_axis: axis over which attention is applied.
    segmentation_mask: bool: if true use equality on cartesian product rather
      than outer product for constructing segmentation masks.
  Returns:
    The padding mask for attention weights.
  """
  assert query_shape[0] == key_shape[0]
  assert len(query_shape) == len(key_shape)

  ndim = len(key_shape)
  if attention_axis is None:
    attention_axis = tuple(range(1, ndim - 2))
  assert isinstance(attention_axis, tuple)
  for ax in attention_axis:
    if not (ndim >= 3 and 1 <= ax < ndim - 2):
      raise ValueError(
          'Attention axis must be between the batch axis and the last-two axes.'
      )

  mask_shape_final = (query_shape[0], 1)  #  batch_size, 1 (for all heads)s
  for ax in attention_axis:
    mask_shape_final += (query_shape[ax],)
  for ax in attention_axis:
    mask_shape_final += (key_shape[ax],)

  padding_mask_query = padding_mask_query[..., None]
  padding_mask_key = padding_mask_key[..., None]
  perm = (0,) + tuple(onp.flip(onp.arange(padding_mask_key.ndim)))[:-1]
  if segmentation_mask:
    mask = jnp.equal(padding_mask_query, padding_mask_key.transpose(perm))
  else:
    mask = jnp.multiply(padding_mask_query, padding_mask_key.transpose(perm))

  mask = mask.reshape(mask_shape_final)
  mask = jax.lax.convert_element_type(mask, jnp.float32)
  return mask


def _make_causal_mask(key, attention_axis=None, self_mask=False):
  """The `flax.nn` module is Deprecated, use `flax.linen` instead. 
  Learn more and find an upgrade guide at 
  https://github.com/google/flax/blob/master/flax/linen/README.md"
  Makes a causal mask, to be used for masking out the future for attention.

  In case of 1d inputs (i.e., `[bs, len, features]`, the attention weights will
  be `[bs, len, len]` and this function makes a square matrix [len, len] with
  zeros in upper triangle and ones in lower triangle.

  Args:
    key: shape of the key, which is equal to the shape of value and is
      assumed to be equal to the shape of the query (since this is used in
      self-attention when decoding).
    attention_axis: axis over which attention is applied.
    self_mask: if mask out the diagonal or not.

  Returns:
    A causal mask to be used to mask out future positions.
  """
  if attention_axis is None:
    attention_axis = tuple(range(1, key.ndim - 2))
  assert isinstance(attention_axis, tuple)
  for ax in attention_axis:
    if not (key.ndim >= 3 and 1 <= ax < key.ndim - 2):
      raise ValueError(
          'Attention axis must be between the batch axis and the last-two axes.'
      )

  mask_shape = tuple([1] * (key.ndim - len(attention_axis) - 1))
  mask_shape_final = mask_shape
  for _ in range(2):
    flatten_dim = 1
    for ax in attention_axis:
      mask_shape_final += (key.shape[ax],)
      flatten_dim *= key.shape[ax]
    mask_shape += (flatten_dim,)

  def tri(n, m, k=0):
    # Tie in the key to avoid the mask becoming a constant.
    # This way XLA can construct the mask during computation and fuse it
    # with the attention ops.
    x = jax.lax.tie_in(key, jnp.arange(n, dtype=jnp.int32))
    y = jax.lax.tie_in(key, jnp.arange(m, dtype=jnp.int32))
    mask = jax.lax.ge(
        (jax.lax.broadcast_in_dim(x, shape=(n, m), broadcast_dimensions=(0,))) + k,
        jax.lax.broadcast(y, [n]))
    return mask

  k = -1 if self_mask else 0
  mask = tri(*mask_shape[-2:], k=k).reshape(mask_shape_final)
  return mask

class ProbedTransformerEncoderLayer(nn.Module):

  def apply(self,
            inputs,
            mask,
            activation_fn=flax.nn.relu,
            num_heads=8,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies one transformer encoder layer.

    Args:
      inputs: The inputs to the transformer, a
        [batch_size, max_num_data_points, data_dim] tensor.
      mask: The mask for the inputs indicating which elements are padding. A
        tensor of shape [batch_size, max_num_data_points].
      activation_fn: The activation function to use, defaults to relu.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_num_data_points, data_dim] tensor of outputs.
    """
    value_dim = inputs.shape[-1]

    attn_outs, query, key, attn_weights = SelfAttention(
        inputs_q=inputs,
        num_heads=num_heads,
        qkv_features=value_dim,
        padding_mask=mask,
        kernel_init=weight_init)

    attn_outs = inputs + attn_outs
    
    attn_outs = normalize(attn_outs, normalization)

    out1 = activation_fn(flax.nn.Dense(attn_outs,
                                       features=value_dim,
                                       kernel_init=weight_init))
    out2 = flax.nn.Dense(out1,
                         features=value_dim,
                         kernel_init=weight_init)

    outs = attn_outs + out2
    return normalize(outs, normalization), query, key, attn_weights


class ProbedTransformerEncoderStack(nn.Module):

  def apply(self,
            inputs,
            mask,
            num_probe_outs,
            batch_size=2,
            num_encoders=6,
            num_heads=8,
            value_dim=128,
            activation_fn=flax.nn.relu,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_normal()):
    """Applies a stack of transformer encoder layers.

    Args:
      inputs: The inputs to the transformer, a
        [batch_size, max_num_data_points, data_dim] tensor.
      mask: The mask for the inputs indicating which elements are padding. A
        tensor of shape [batch_size, max_num_data_points].
      num_encoders: The number of encoder layers in the stack.
      num_heads: The number of heads to use for self-attention, defaults to 8.
      value_dim: The dimension of the transformer's keys, values, and queries.
      activation_fn: The activation function to use, defaults to relu.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: A [batch_size, max_num_data_points, value_dim] tensor of outputs.
    """
    inputs = flax.nn.Dense(inputs, features=value_dim, kernel_init=weight_init)
    reps = [inputs]
    
    probe = flax.nn.DenseGeneral.partial(features=num_probe_outs, batch_dims=(0,))
    probe_inputs = jax.lax.stop_gradient(inputs)
    probe_outs = [probe(probe_inputs, name="probe_0")]
    queries = []
    keys = []
    attn_weights = []
    for i in range(num_encoders):
      inputs, query, key, weights = ProbedTransformerEncoderLayer(inputs,
                                       mask,
                                       activation_fn=activation_fn,
                                       num_heads=num_heads,
                                       normalization=normalization,
                                       weight_init=weight_init,
                                       name="TransformerEncoderLayer_%d" % (i+1))
      queries.append(query)
      keys.append(key)
      attn_weights.append(weights)
      reps.append(inputs)
      probe_inputs = jax.lax.stop_gradient(inputs)
      probe_out = probe(probe_inputs, name="probe_%d" % (i+1))
      probe_outs.append(probe_out)
    return inputs, probe_outs, reps, queries, keys, attn_weights


class ProbedUnconditionalEncoderDecoderTransformer(nn.Module):

  def apply(self,
            inputs,
            input_lengths,
            target_lengths,
            probe_out_dim=None,
            batch_size=2,
            targets=None,
            target_dim=32,
            max_target_length=100,
            num_heads=8,
            num_encoders=6,
            num_decoders=6,
            qkv_dim=512,
            activation_fn=flax.nn.relu,
            normalization=None,
            weight_init=jax.nn.initializers.xavier_uniform()):
    """Applies Transformer model on the inputs.

    Args:
      inputs: input data, [batch_size, max_num_data_points, data_dim].
      input_lengths: A [batch_size] vector containing the number of samples
        in each batch element.
      target_lengths: Unused.
      targets: Unused.
      target_dim: The length of each output vector.
      max_target_length: An int at least as large as the largest element of
        target_lengths, used for determining output shapes.
      num_heads: The number of heads for the self attention.
      num_encoders: The number of transformer encoder layers.
      num_decoders: The number of transformer decoder layers.
      qkv_dim: The dimension of the query/key/value.
      activation_fn: The activation function to use, defaults to relu.
      weight_init: An initializer for the encoder weights.
    Returns:
      outs: The transformer output, a tensor of shape
        [batch_size, max_target_length, target_dim].
    """
    max_input_length = inputs.shape[1]
    input_mask = util.make_mask(input_lengths, max_input_length)

    encoder_hs, probe_outs, reps, queries, keys, attn_weights = ProbedTransformerEncoderStack(
        inputs,
        input_mask,
        probe_out_dim,
        batch_size=batch_size,
        num_encoders=num_encoders,
        num_heads=num_heads,
        value_dim=qkv_dim,
        normalization=normalization,
        weight_init=weight_init,
        name="TransformerEncoderStack_0")
    # average over data dimension, resulting in [batch_size, data_dim]
    encoder_out = jnp.mean(encoder_hs, axis=1)

    decoder_out = flax.nn.Dense(encoder_out, features=max_target_length*64, 
        kernel_init=weight_init, name="Dense_1")

    for i in range(num_decoders):
      layer_out = activation_fn(flax.nn.Dense(decoder_out,
                                       features=max_target_length*64,
                                       kernel_init=weight_init,
                                       name="Dense_%d" % (i*4 + 2)))
      layer_out = normalize(layer_out, normalization, name="LayerNorm_%d" % (i*4 + 3))

      layer_out = flax.nn.Dense(layer_out,
                                features=max_target_length*64,
                                kernel_init=weight_init,
                                name="Dense_%d" % (i*4 + 4))
      layer_out = activation_fn(layer_out)
      decoder_out = normalize(layer_out + decoder_out, normalization, name="LayerNorm_%d" % (i*4 +
        5))


    # dense layer to arrive at [batch_size, target_length, target_dim]
    out = flax.nn.Dense(
        decoder_out,
        features=target_dim*max_target_length,
        kernel_init=weight_init,
        name="Dense_%d" % (num_decoders*4+2))
    out = jnp.reshape(out, [batch_size, max_target_length, target_dim])

    return out, jnp.array(probe_outs), jnp.array(reps), jnp.array(queries), jnp.array(keys), jnp.array(attn_weights)
