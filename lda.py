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

"""Code for sampling from the LDA generative model."""
import jax
from jax import jit
from jax import vmap
import jax.numpy as jnp
import jax.scipy as jscipy
from functools import partial

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def doc_topic_em_inference(document_words, log_topic_params, num_topics, tol):
  
  # [num_topics, vocab_size]
  topic_word_log_probs = log_topic_params[:, document_words]

  def em_step(carry):
    # [num_topics, vocab_size]
    prev_log_doc_topic_params, prev_likelihood, _, _ = carry
    pseudo_obs = topic_word_log_probs + prev_log_doc_topic_params[:, jnp.newaxis]
    pseudo_obs -= jscipy.special.logsumexp(pseudo_obs, axis=0, keepdims=True)
    # [num_topics]
    logits = jscipy.special.logsumexp(pseudo_obs, axis=1)
    new_log_doc_topic_params = jax.nn.log_softmax(logits)
    # Compute the objective
    log_zs = jnp.exp(pseudo_obs)
    new_likelihood = jnp.sum(log_zs*new_log_doc_topic_params[:, jnp.newaxis])
    return (new_log_doc_topic_params, new_likelihood, 
            prev_log_doc_topic_params, prev_likelihood)
  
  def em_stopping_cond(carry):
    cur_params, cur_likelihood, prev_params, prev_likelihood = carry
    return jnp.abs(cur_likelihood - prev_likelihood) >= tol

  init_params = jnp.full([num_topics], -jnp.log(num_topics))
  init_likelihood = -jnp.inf
  outs = jax.lax.while_loop(em_stopping_cond, em_step, 
                            (init_params, init_likelihood, 
                             jnp.ones_like(init_params), 0.))
  out_params, out_likelihood = outs[0:2]
  return out_params

def document_log_prob(document_words, document_log_topic_probs, log_topic_params):
  # log p(w) = log \sum_topic p(word|topic)p(topic)
  # Obtain matrix that is [num_topics, document_length] which is log prob(word | topic)
  word_topic_log_probs = log_topic_params[:, document_words]
  # add the log prob of each topic to each row in the matrix
  word_topic_joint_log_probs = word_topic_log_probs + document_log_topic_probs[:, jnp.newaxis]
  # log sum exp each column, giving the log prob of each word
  word_log_probs = jscipy.special.logsumexp(word_topic_joint_log_probs, axis=0)
  # sum the remaining vector, giving the sum of all log probs of words in the document
  doc_log_prob = jnp.sum(word_log_probs)
  return doc_log_prob

def perplexity(documents_words, documents_log_topic_probs, log_topic_params):
  num_docs, doc_length = documents_words.shape
  document_log_probs = vmap(document_log_prob, in_axes=(0, 0, None))(
      documents_words, documents_log_topic_probs, log_topic_params)
  perplexity = jnp.exp(- jnp.sum(document_log_probs) / (num_docs * doc_length))
  return perplexity

@partial(jit, static_argnums=(2,3))
def topic_param_perplexity(documents_words, log_topic_params, num_topics, em_tol):
  docs_log_topic_probs = vmap(doc_topic_em_inference, in_axes=(0, None, None, None))(
      documents_words, log_topic_params, num_topics, em_tol)
  return perplexity(documents_words, docs_log_topic_probs, log_topic_params)

def sample_log_dirichlet(key, alpha, shape=()):
  gamma_shape = tuple(list(shape) + [alpha.shape[0]])
  gammas = jax.random.gamma(key, alpha, shape=gamma_shape)
  log_gammas = jnp.log(gammas)
  log_probs = jax.nn.log_softmax(log_gammas, axis=-1)
  return log_probs

@partial(jit, static_argnums=(1,))
def sample_params(key, num_docs, doc_topic_alpha, topic_word_alpha):
  """Samples the parameters for LDA, the topic/word dists and doc/topic dists.

  Args:
    key: A JAX PRNG key.
    num_docs: The number of documents to sample.
    doc_topic_alpha: The parameter alpha for the dirichlet prior over the
      document-topic categorical distributions. Should be a vector of shape
      [num_topics].
    topic_word_alpha: The alpha parameter for the dirichlet prior over the
      topic-word categorical distributions. Should be a vector of shape
        [vocab_size].
  Returns:
    log_topic_params: A [num_topics, vocab_size] matrix containing parameters
      for each topic's categorical distribution over words.
    log_doc_params: A [num_docs, num_topics] matrix containing parameters for
      each document's categorical distribution over topics.
  """
  # Each topic is a multinomial distribution over words.
  # The topic matrix is a [num_topics, vocab_size] set of parameters for
  # num_topics different multinomial distributions over the vocabulary.
  num_topics = doc_topic_alpha.shape[0]
  k1, k2 = jax.random.split(key)
  log_topic_params = sample_log_dirichlet(k1, topic_word_alpha, shape=[num_topics])
  # Each document is a multinomial distribution over topics.
  # The document matrix is a [num_documents, num_topics] set of parameters for
  # num_documents different multinomial distributions over the set of topics.
  log_doc_params = sample_log_dirichlet(k2, doc_topic_alpha, shape=[num_docs])
  return log_topic_params, log_doc_params


@partial(jit, static_argnums=3)
def sample_docs(key, log_topic_params, log_doc_params, doc_length):
  """Samples documents given parameters.

  Args:
    key: A JAX PRNG key.
    log_topic_params: A [num_topics, vocab_size] matrix containing parameters
      for each topic's categorical distribution over words.
    log_doc_params: A [num_docs, num_topics] matrix containing parameters for
      each document's categorical distribution over topics.
    doc_length: The length of each document, a python int.
  Returns:
    doc_words: A [num_documents, doc_length] matrix containing the indices of
      each word in each document. Each index will be in [0, vocab_size).
  """
  num_documents, _ = log_doc_params.shape
  k1, k2 = jax.random.split(key)
  # Sample the sequence of topics for each document, a
  # [num_documents, doc_length] matrix of topic indices.
  doc_topics = jax.random.categorical(k1, log_doc_params, shape=(doc_length, num_documents)).T
  # Sample the sequence of words for each document, a
  # [doc_length, num_documents] matrix of word indices.
  sample_doc = lambda a: jax.random.categorical(a[1], log_topic_params[a[0]])
  keys = jax.random.split(k2, num=num_documents)
  doc_words = jax.lax.map(sample_doc, (doc_topics, keys))
  return doc_words, doc_topics


@partial(jit, static_argnums=(1, 2, 3, 4))
def sample_lda(key, num_docs, num_topics, vocab_size, doc_length):
  """Samples documents and parameters from LDA using default prior parameters.

  Samples from LDA assuming that each element of doc_topic_alpha is 1/num_topics
  and each element of topic_word_alpha is 1/vocab_size.

  Args:
    key: A JAX PRNG key.
    num_docs: The number of documents to sample, a python int.
    num_topics: The number of latent topics, a python int.
    vocab_size: The number of possible words, a python int.
    doc_length: The length of each document, a python int.
  Returns:
    doc_words: A [num_documents, doc_length] matrix containing the indices of
      each word in each document. Each index will be in [0, vocab_size).
    log_topic_params: A [num_topics, vocab_size] matrix containing parameters
      for each topic's categorical distribution over words.
    log_doc_params: A [num_docs, num_topics] matrix containing parameters for
      each document's categorical distribution over topics.
  """
  key1, key2 = jax.random.split(key)
  log_topic_params, log_doc_params = sample_params(
      key1, num_docs, jnp.ones([num_topics]), jnp.ones([vocab_size]))
  doc_words, doc_topics = sample_docs(key2, log_topic_params, log_doc_params, doc_length)
  return doc_words, doc_topics, log_topic_params, log_doc_params

def gibbs_word_topic_conditional(
    index,
    doc_words,
    doc_topics,
    log_topic_params,
    doc_topic_alpha,
    temperature):
  """Compute the distribution over a specific topic given other topics and all words.

  Args:
    index: The index of the topic to compute the distribution for.
    doc_words: An integer vector of shape [doc_length], the set of words in the document.
    doc_topics: An integer vector of shape [doc_length], the set of topics associated with
      each word in the document. The topic at 'index' is not used.
    log_topic_params: A [num_topics, vocab_size] matrix representing the parameters of each topic.
    doc_topic_alpha: The concentration parameter for the Dirichlet prior over the document
      topic distributions. Must be a vector of shape [num_topics].
  
  Returns:
    log_probs: A float vector of shape [num_topics], the log probabilities of the distsribution
      over the topics of the index-th word given the other topics, all words, and the topic 
      parameters.
  """
  num_topics = doc_topic_alpha.shape[0]
  num_words = doc_words.shape[0]
  log_p_word = log_topic_params[:, doc_words[index]]
  topic_counts = jnp.sum(
      jax.nn.one_hot(doc_topics, num_classes=num_topics), axis=0)
  topic_counts = topic_counts - jax.nn.one_hot(doc_topics[index],
                                               num_classes=num_topics)
  p_topic = (topic_counts + doc_topic_alpha)/(
      num_words - 1 + jnp.sum(doc_topic_alpha))
  log_probs = temperature * log_p_word + jnp.log(p_topic)
  return log_probs

def gibbs_pass(key, doc_words, doc_topics, log_topic_params, doc_topic_alpha, temperature):
  """Runs one pass of Gibbs sampling.

  Runs one full pass of Gibbs sampling on the topics of a document, sampling each topic
  once.

  Args:
    key: A JAX PRNG key.
    doc_words: An integer vector of shape [doc_length], the words in the document.
    doc_topics: An integer vector of shape [doc_length], the topics associated with each word
      in the document.
    log_topic_params: A [num_topics, vocab_size] matrix representing each topics distribution over
      words.
    doc_topic_alpha: The concentration parameter for the dirichlet prior on the document-topic
      distribution.
    temperature: The annealing temperature.

  Returns:
    doc_topics: A set of topics for each word in the document after resampling each topic once
      using Gibbs sampling.
  """
  def gibbs_step(i, carry):
    key, doc_topics = carry
    log_p_topics = gibbs_word_topic_conditional(
        i, doc_words, doc_topics, log_topic_params, doc_topic_alpha, temperature)
    key, new_key = jax.random.split(key)
    topic = tfd.Categorical(logits=log_p_topics).sample(seed=key)
    new_doc_topics = jax.ops.index_update(doc_topics, i, topic)
    return (new_key, new_doc_topics)

  _, doc_topics = jax.lax.fori_loop(
      0, doc_topics.shape[0], gibbs_step, (key, doc_topics))
  return doc_topics

def lda_ais(key, doc_words, log_topic_params, doc_topic_alpha, num_dists, num_samples):
  """Computes an AIS estimate of the log marginal probability of a document.

  Args:
    key: A JAX PRNG key.
    doc_words: An integer vector of shape [doc_length], the words in the document.
    log_topic_params: A [num_topics, vocab_size] matrix representing each topics distribution over
      words.
    doc_topic_alpha: The concentration parameter for the dirichlet prior on the document-topic
      distribution.
    num_dists: The number of intermediate distributions.
    num_samples: The number of samples to average the estimate over.
  Returns:
    log_Z_hat: An estimate of the log marginal probability of the document.
  """
  temps = jnp.linspace(0, 1, num=num_dists)
  doc_length = doc_words.shape[0]

  def sample_prior(key):
    key1, key2 = jax.random.split(key)
    topic_dist = tfd.Dirichlet(doc_topic_alpha).sample(seed=key1)
    doc_topics = tfd.Categorical(logits=topic_dist).sample(
        seed=key2, sample_shape=[doc_length])
    return doc_topics

  def sample_chain_step(carry, i):
    key, doc_topics = carry
    new_key, key = jax.random.split(key)
    new_doc_topics = gibbs_pass(
        key, doc_words, doc_topics, log_topic_params, doc_topic_alpha, temps[i])
    return (new_key, new_doc_topics), new_doc_topics

  def sample_chain(key, init_doc_topics):
    _, docs = jax.lax.scan(
        sample_chain_step, (key, init_doc_topics), jnp.arange(1, num_dists))
    return docs

  # [num_samples, doc_length]
  init_doc_topics = vmap(sample_prior)(jax.random.split(key, num=num_samples))
  # [num_samples, num_dists - 1, doc_length]
  topic_chains = vmap(sample_chain)(
      jax.random.split(key, num=num_samples), init_doc_topics)
  # [num_topics, doc_length]
  word_topic_log_probs = log_topic_params[:, doc_words]
  # [num_samples, num_dists - 1, doc_length]
  log_probs = vmap(lambda t, w: w[t], in_axes=(2,1), out_axes=2)(
      topic_chains, word_topic_log_probs)
  # [num_samples, num_dists - 1]
  doc_log_probs = jnp.sum(log_probs, axis=2)
  # [num_dists - 1]
  temp_deltas = jnp.diff(temps)
  # [num_samples]
  annealed_log_ps =  jnp.sum(
      doc_log_probs * temp_deltas[jnp.newaxis, :], axis=1)
  return jscipy.special.logsumexp(annealed_log_ps)
