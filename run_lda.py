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

"""Runner for LDA transformer experiments.
"""
import os

import lda
import lda_models
import train

from absl import app
from absl import flags
import jax
from jax import vmap
from jax.config import config
import jax.experimental.optimizers
import jax.numpy as jnp
import numpy as onp

import sklearn
from sklearn import decomposition
sklearn_lda = decomposition.LatentDirichletAllocation

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


flags.DEFINE_enum("model", "topic_word", ["topic_word"],
                  "The model to run, either training topic_word model or "
                  "doc_topic model.")
flags.DEFINE_integer("num_encoders", 6,
                     "Number of encoder modules in the transformer.")
flags.DEFINE_integer("num_decoders", 6,
                     "Number of decoder modules in the transformer.")
flags.DEFINE_integer("num_heads", 8,
                     "Number of attention heads in the transformer.")
flags.DEFINE_integer("key_dim", 32,
                     "The dimension of the keys in the transformer.")
flags.DEFINE_integer("value_dim_per_head", 32,
                     "The dimension of the values in the transformer for "
                     "each head.")
flags.DEFINE_integer("embedding_dim", 256,
                     "The dimension of the word embedding vectors.")
flags.DEFINE_integer("num_docs", 100,
                     "The number of documents per dataset.")
flags.DEFINE_integer("num_topics", 5,
                     "The number of topics per dataset.")
flags.DEFINE_integer("vocab_size", 1000,
                     "The size of the vocabulary.")
flags.DEFINE_integer("doc_length", 25,
                     "The number of words in each document.")
flags.DEFINE_boolean("parallel", True,
                     "If possible, train in parallel across devices.")
flags.DEFINE_integer("batch_size", 64,
                     "The batch size.")
flags.DEFINE_integer("eval_batch_size", 256,
                     "The batch size for evaluation.")
flags.DEFINE_integer("num_steps", int(1e6),
                     "The number of steps to train for.")
flags.DEFINE_float("lr", 1e-3,
                   "The learning rate for ADAM.")
flags.DEFINE_integer("summarize_every", 100,
                     "Number of steps between summaries.")
flags.DEFINE_integer("checkpoint_every", 5000,
                     "Number of steps between checkpoints.")
flags.DEFINE_boolean("clobber_checkpoint", False,
                     "If true, remove any existing summaries and checkpoints "
                     "in the logdir.")
flags.DEFINE_string("logdir", "/tmp/lda_models",
                    "The directory to put summaries and checkpoints.")
flags.DEFINE_boolean("debug_nans", False,
                     "If true, run in debug mode and fail on nans.")

FLAGS = flags.FLAGS


def make_topic_word_model(key,
                          batch_size=64,
                          num_docs=100,
                          doc_length=50,
                          num_topics=10,
                          vocab_size=1000,
                          num_heads=4,
                          num_encoders=2,
                          num_decoders=2,
                          qkv_dim=128,
                          embedding_dim=128,
                          test_percent=0.2):
  key, subkey = jax.random.split(key)
  model = lda_models.LDATopicWordInferenceMachine.partial(
      num_topics=num_topics, vocab_size=vocab_size, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=qkv_dim,
      embedding_dim=embedding_dim)

  _, params = model.init_by_shape(
      subkey, [((batch_size, num_docs, doc_length), jnp.int32),
               ((batch_size, num_topics, vocab_size), jnp.float32)])

  def sample_batch(key):
    return vmap(lda.sample_lda, in_axes=(0, None, None, None, None))(
        jax.random.split(key, num=batch_size), num_docs, num_topics,
        vocab_size, doc_length)

  def loss(params, key):
    doc_words, _, log_topic_params, _ = sample_batch(key)
    losses = model.loss(params, subkey, doc_words, log_topic_params)
    return jnp.mean(losses)

  def dataset_ais(key, docs_words, log_topic_params, doc_topic_alpha, num_dists, num_samples):
    docs_log_ps = vmap(lda.lda_ais, in_axes=(0, 0, None, None, None, None))(
        jax.random.split(key, num=num_docs), docs_words, log_topic_params, doc_topic_alpha, 
        num_dists, num_samples)
    return jnp.mean(docs_log_ps)
  
  batch_ais = vmap(dataset_ais, in_axes=(0, 0, 0, None, None, None))

  batch_perplexity = vmap(lda.topic_param_perplexity, in_axes=(0, 0, None, None))

  def fit_sklearn_lda(docs_words, num_topics, vocab_size):
    # [batch_size, num_docs, doc_length, vocab_size] -> [batch_size, num_docs, vocab_size]
    docs_word_counts = jnp.sum(jax.nn.one_hot(docs_words, num_classes=vocab_size), axis=2)
    out_params = []
    for i in range(batch_size):
      out = sklearn_lda(
              n_components=num_topics, 
              doc_topic_prior=1.,
              topic_word_prior=1.).fit(docs_word_counts[i])
      topic_logits = jnp.log(out.components_)
      normalized_log_topic_params = jax.nn.log_softmax(topic_logits, axis=1)
      out_params.append(normalized_log_topic_params)
    return jnp.array(out_params)

  def summarize(writer, step, params, key):
    em_tol = 1e-4
    k1, k2 = jax.random.split(key)
    # [batch_size, num_documents, doc_length]
    doc_words, doc_topics, true_log_topic_params, _ = sample_batch(k1)
    # [batch_size, num_topics, vocab_size]
    pred_log_topic_params = model.call(params, doc_words, None)
    sklearn_log_topic_params = fit_sklearn_lda(doc_words, num_topics, vocab_size)

    pred_perplexity = batch_perplexity(doc_words, pred_log_topic_params, num_topics, em_tol)
    sklearn_perplexity = batch_perplexity(doc_words, sklearn_log_topic_params, num_topics, em_tol)
    true_perplexity = batch_perplexity(doc_words, true_log_topic_params, num_topics, em_tol)

    writer.scalar("perplexity", jnp.mean(pred_perplexity), step=step)
    writer.scalar("perplexity_sklearn", jnp.mean(sklearn_perplexity), step=step)
    writer.scalar("perplexity_true_params", jnp.mean(true_perplexity), step=step)
    print("perplexity: %0.3f" % jnp.mean(pred_perplexity))
    print("sklearn perplexity: %0.3f" % jnp.mean(sklearn_perplexity))
    print("perplexity true params: %0.3f" % jnp.mean(true_perplexity))

  return params, loss, summarize


def make_logdir(config):
  basedir = config.logdir
  exp_dir = (
      "%s_ndocs_%d_ntop_%d_voc_%d_embd_%d_nheads_%d_nencoders_%d_ndecoders_%d"
      % (config.model, config.num_docs, config.num_topics, config.vocab_size,
         config.embedding_dim, config.num_heads, config.num_encoders,
         config.num_decoders))
  return os.path.join(basedir, exp_dir)


def main(unused_argv):
  config.update("jax_debug_nans", False)
  key = jax.random.PRNGKey(0)
  k1, k2 = jax.random.split(key)

  if FLAGS.parallel and train.can_train_parallel():
    assert FLAGS.batch_size % jax.local_device_count(
    ) == 0, "Device count must evenly divide batch_size"
    FLAGS.batch_size = int(FLAGS.batch_size / jax.local_device_count())

  init_params, loss_fn, summarize_fn = make_topic_word_model(
          k1,
          batch_size=FLAGS.batch_size,
          num_docs=FLAGS.num_docs,
          doc_length=FLAGS.doc_length,
          num_topics=FLAGS.num_topics,
          vocab_size=FLAGS.vocab_size,
          num_heads=FLAGS.num_heads,
          num_encoders=FLAGS.num_encoders,
          num_decoders=FLAGS.num_decoders,
          qkv_dim=FLAGS.value_dim_per_head*FLAGS.num_heads,
          embedding_dim=FLAGS.embedding_dim)
  train.train_loop(
          k2, init_params, loss_fn,
          parallel=FLAGS.parallel,
          lr=FLAGS.lr,
          num_steps=FLAGS.num_steps,
          summarize_fn=summarize_fn,
          summarize_every=FLAGS.summarize_every,
          checkpoint_every=FLAGS.checkpoint_every,
          clobber_checkpoint=FLAGS.clobber_checkpoint,
          logdir=make_logdir(FLAGS))


if __name__ == "__main__":
  app.run(main)
