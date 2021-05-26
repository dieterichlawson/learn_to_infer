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

import flax
from flax import nn
import jax
import jax.numpy as jnp
import jax.random
from jax import vmap
import jax.experimental
import jax.experimental.host_callback as hcb

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
    batch_size = inputs.shape[0]
    max_input_length = inputs.shape[1]
    input_mask = util.make_mask(input_lengths, max_input_length)

    encoder_hs = TransformerEncoderStack(inputs,
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

    for i in range(num_encoders):
      inputs = TransformerEncoderLayer(inputs,
                                       mask,
                                       activation_fn=activation_fn,
                                       num_heads=num_heads,
                                       normalization=normalization,
                                       weight_init=weight_init)
      reps.append(inputs)
      probe_inputs = jax.lax.stop_gradient(inputs)
      probe_out = probe(probe_inputs, name="probe_%d" % (i+1))
      probe_outs.append(probe_out)
    return inputs, probe_outs, reps


class ProbedUnconditionalEncoderDecoderTransformer(nn.Module):

  def apply(self,
            inputs,
            input_lengths,
            target_lengths,
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

    encoder_hs, probe_outs, reps = ProbedTransformerEncoderStack(
        inputs,
        input_mask,
        max_target_length,
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

    return out, jnp.array(probe_outs), jnp.array(reps)
