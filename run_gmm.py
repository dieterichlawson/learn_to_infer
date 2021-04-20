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

"""Runner for transformer GMM experiments.
"""
import os

import sample_gmm
import gmm_eval
import gmm_models
import plotting
import train

from absl import app
from absl import flags

import jax
from jax.config import config
import jax.numpy as jnp

import scipy as oscipy
import matplotlib.pyplot as plt


flags.DEFINE_enum("model_name", "mean_scale_weight",
                  #["mean", "mean_scale", "mean_scale_weight"],
                  ["mean_scale_weight"],
                  "Model to run")
flags.DEFINE_integer("num_encoders", 6,
                     "Number of encoder modules in the transformer.")
flags.DEFINE_integer("num_decoders", 6,
                     "Number of decoder modules in the transformer.")
flags.DEFINE_integer("num_heads", 8,
                     "Number of attention heads in the transformer.")
flags.DEFINE_integer("key_dim", 32,
                     "The dimension of the keys in the transformer.")
flags.DEFINE_integer("value_dim_per_head", 32,
                     "The dimension of the values in the transformer "
                     "for each head.")
flags.DEFINE_integer("data_dim", 2,
                     "The dimension of the points to cluster.")
flags.DEFINE_integer("k", None,
                     "The number of modes in the data. If provided, overrides min_k and max_k.")
flags.DEFINE_integer("min_k", 2,
                     "The minimum number of modes in the data.")
flags.DEFINE_integer("max_k", 10,
                     "The maximum number of modes in the data.")
flags.DEFINE_integer("data_points_per_mode", 25,
                     "Number of data points to include per mode in the data.")
flags.DEFINE_integer("cov_dof", None,
                     "Degrees of freedom in sampling the random covariances.")
flags.DEFINE_enum("cov_prior", "inv_wishart",
                  ["inv_wishart", "wishart"],
                  "The prior to use for the covariance matrix.")
flags.DEFINE_float("mode_var", 1.,
                   "The variance of the modes in the GMM used when "
                   "not sampling.")
flags.DEFINE_float("dist_multiplier", .95,
                   "Confidence interval that will be nonoverlapping when sampling the meaans")
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
flags.DEFINE_string("logdir", "/tmp/transformer",
                    "The directory to put summaries and checkpoints.")
flags.DEFINE_boolean("debug_nans", False,
                     "If true, run in debug mode and fail on nans.")
flags.DEFINE_boolean("plot_sklearn_comparison", False,
                     "If true, generate the sklearn clustering comparison "
                     "plots.")
flags.DEFINE_enum("normalization", 
                  "no_norm", ["no_norm", "layer_norm", "batch_norm"],
                  "Type of normalization to use")
flags.DEFINE_string("tag", "",
                    "String to append to the logdir.")

FLAGS = flags.FLAGS


def make_model(key,
               model_name="mean",
               num_encoders=4,
               num_decoders=4,
               num_heads=8,
               value_dim=128,
               data_points_per_mode=25,
               max_k=10,
               data_dim=2,
               normalization="no_norm"):
  class_dict = {
      #"mean": gmm_models.MeanInferenceMachine,
      #"mean_scale": gmm_models.MeanScaleInferenceMachine,
      "mean_scale_weight": gmm_models.MeanScaleWeightInferenceMachine}

  model = class_dict[model_name](
      data_dim=data_dim, max_k=max_k,
      max_num_data_points=max_k*data_points_per_mode, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=value_dim,
      normalization=normalization)
  params = model.init_params(key)

  return model, params


def make_loss(model,
              model_name="mean",
              min_k=2,
              max_k=10,
              data_points_per_mode=25,
              cov_dof=10,
              cov_prior="inv_wishart",
              mode_var=1.,
              dist_mult=2.,
              data_dim=2,
              batch_size=128):

  def sample_train_batch(key):
    return sample_gmm.sample_batch_random_ks(
        key, model_name, batch_size, min_k, max_k, data_points_per_mode,
        data_dim, mode_var, cov_dof, cov_prior, dist_mult)

  def loss(params, key):
    key, subkey = jax.random.split(key)
    xs, _, ks, mog_params = sample_train_batch(key)
    losses = model.loss(
        params, xs, ks*data_points_per_mode, mog_params, ks, subkey)
    return jnp.mean(losses)

  return loss


def make_summarize(
    model,
    model_name="mean",
    min_k=2,
    max_k=10,
    data_points_per_mode=25,
    cov_dof=10,
    cov_prior="inv_wishart",
    dist_mult=2.,
    data_dim=2,
    mode_var=1.,
    eval_batch_size=256):

  def sample_eval_batch(key):
    return sample_gmm.sample_batch_random_ks(
        key, model_name, eval_batch_size, min_k, max_k, data_points_per_mode,
        data_dim, mode_var, cov_dof, cov_prior, dist_mult)

  sample_eval_batch = jax.jit(sample_eval_batch)

  def sample_single_gmm(key, num_modes):
    xs, cs, params = sample_gmm.sample_batch_fixed_ks(
        key, model_name, jnp.array([num_modes]), max_k, data_points_per_mode,
        data_dim, mode_var, cov_dof, cov_prior, dist_mult)

    return xs[0], cs[0], (params[0][0], params[1][0], params[2][0])

  def model_classify(params, inputs, ks):
    return gmm_models.classify_with_defaults(
        model, params, inputs, eval_batch_size, ks*data_points_per_mode, ks,
        max_k, jnp.eye(data_dim)*mode_var)

  def sample_and_classify_eval_batch(key, params):
    xs, cs, ks, true_gmm_params = sample_eval_batch(key)
    tfmr_cs, tfmr_gmm_params = model_classify(params, xs, ks)
    return xs, cs, ks, true_gmm_params, tfmr_cs, tfmr_gmm_params

  def sample_and_classify_single_gmm(key, params, num_modes):
    xs, cs, gmm_params = sample_single_gmm(key, num_modes)
    tfmr_cs, tfmr_gmm_params = model_classify(
        params, xs[jnp.newaxis], jnp.array([num_modes]))
    return xs, cs, gmm_params, tfmr_cs, tfmr_gmm_params

  sample_and_classify_single_gmm = jax.jit(sample_and_classify_single_gmm)

  def summarize_baselines(writer, step, key):
    key, subkey = jax.random.split(key)
    xs, cs, ks, _ = sample_eval_batch(subkey)
    em_metrics, srbf_metrics, agg_metrics = gmm_eval.compute_masked_baseline_metrics(
        xs, cs, ks, ks*data_points_per_mode)
    # EM
    writer.scalar("em/pairwise_acc", em_metrics[0], step=step)
    print("em pairwise acc: %0.3f" % em_metrics[0])
    writer.scalar("em/pairwise_f1", em_metrics[1], step=step)
    print("em pairwise f1: %0.3f" % em_metrics[1])
    writer.scalar("em/avg_ll", em_metrics[2], step=step)
    print("em avg ll: %0.3f" % em_metrics[2])
    # Spectral RBF
    writer.scalar("spectral_rbf/pairwise_acc", srbf_metrics[0], step=step)
    print("spectral rbf pairwise acc: %0.3f" % srbf_metrics[0])
    writer.scalar("spectral_rbf/pairwise_f1", srbf_metrics[1], step=step)
    print("spectral rbf pairwise f1: %0.3f" % srbf_metrics[1])
    # Agglomerative Clustering
    writer.scalar("agglomerative/pairwise_acc", agg_metrics[0], step=step)
    print("agglomerative pairwise acc: %0.3f" % agg_metrics[0])
    writer.scalar("agglomerative/pairwise_f1", agg_metrics[1], step=step)
    print("agglomerative pairwise f1: %0.3f" % agg_metrics[1])

  def plot_params(num_modes, num_data_points, writer, step, params, key):
    outs = sample_and_classify_single_gmm(key, params, num_modes)
    xs, true_cs, true_params, pred_cs, pred_params = outs
    pred_cs = pred_cs[0]
    pred_params = (pred_params[0][0], pred_params[1][0], pred_params[2][0])
    em_cs, em_params = plotting.fit_em(xs, num_modes)
    fig = plotting.plot_gmms(
        xs, num_modes, true_cs, true_params, pred_cs, pred_params, em_cs,
        em_params)
    plot_img = plotting.plot_to_numpy_image(plt)
    writer.image(
        "%d_modes_%d_points" % (num_modes, num_data_points),
        plot_img, step=step)
    plt.close(fig)

  def comparison_inference(params):
    datasets = plotting.make_comparison_gmm_datasets()
    varied_inputs, aniso_inputs, blob_inputs, no_structure_inputs = datasets
    varied_inputs = varied_inputs[0][jnp.newaxis, Ellipsis]
    aniso_inputs = aniso_inputs[0][jnp.newaxis, Ellipsis]
    blob_inputs = blob_inputs[0][jnp.newaxis, Ellipsis]
    no_structure_inputs = no_structure_inputs[0][jnp.newaxis, Ellipsis]

    new_model = gmm_models.MeanScaleWeightInferenceMachine(
        data_dim=data_dim, max_k=max_k,
        max_num_data_points=1500, num_heads=FLAGS.num_heads,
        num_encoders=FLAGS.num_encoders, num_decoders=FLAGS.num_decoders,
        qkv_dim=FLAGS.value_dim_per_head*FLAGS.num_heads)
    varied_cs, varied_params = new_model.classify(
        params, varied_inputs, jnp.array([1500]), jnp.array([3]))
    aniso_cs, aniso_params = new_model.classify(
        params, aniso_inputs, jnp.array([1500]), jnp.array([3]))
    blob_cs, blob_params = new_model.classify(
        params, blob_inputs, jnp.array([1500]), jnp.array([3]))
    no_structure_cs, no_structure_params = new_model.classify(
        params, no_structure_inputs, jnp.array([1500]), jnp.array([3]))
    varied_cs = varied_cs[0]
    aniso_cs = aniso_cs[0]
    blob_cs = blob_cs[0]
    no_structure_cs = no_structure_cs[0]
    varied_params = (varied_params[0][0], varied_params[1][0],
                     varied_params[2][0])
    aniso_params = (aniso_params[0][0], aniso_params[1][0], aniso_params[2][0])
    blob_params = (blob_params[0][0], blob_params[1][0], blob_params[2][0])
    no_structure_params = (no_structure_params[0][0], no_structure_params[1][0],
                           no_structure_params[2][0])
    return (varied_inputs, aniso_inputs, blob_inputs, no_structure_inputs,
            varied_cs, aniso_cs, blob_cs, no_structure_cs,
            varied_params, aniso_params, blob_params, no_structure_params)

  comparison_inference = jax.jit(comparison_inference)

  def plot_comparisons(writer, step, params):
    outs = comparison_inference(params)
    varied_inputs, aniso_inputs, blob_inputs, no_structure_inputs = outs[:4]
    varied_cs, aniso_cs, blob_cs, no_structure_cs = outs[4:8]
    varied_params, aniso_params, blob_params, no_structure_params = outs[8:]
    fig = plotting.plot_comparison_gmm(varied_inputs[0], varied_inputs[1],
                                       varied_cs, varied_params)
    plot_image = plotting.plot_to_numpy_image(plt)
    writer.image("varied", plot_image, step=step)
    plt.close(fig)
    fig = plotting.plot_comparison_gmm(aniso_inputs[0], aniso_inputs[1],
                                       aniso_cs, aniso_params)
    writer.image("aniso", plotting.plot_to_numpy_image(plt), step=step)
    plt.close(fig)
    fig = plotting.plot_comparison_gmm(blob_inputs[0], blob_inputs[1], blob_cs,
                                       blob_params)
    writer.image("blob", plotting.plot_to_numpy_image(plt), step=step)
    plt.close(fig)
    fig = plotting.plot_comparison_gmm(no_structure_inputs[0],
                                       no_structure_inputs[1], no_structure_cs,
                                       no_structure_params)
    writer.image("no_structure", plotting.plot_to_numpy_image(plt), step=step)
    plt.close(fig)

  def compute_metrics(key, params):
    xs, cs, ks, _, tfmr_cs, tfmr_gmm_params  = sample_and_classify_eval_batch(key, params)
    avg_acc, avg_f1, avg_log_marginal = gmm_eval.compute_metrics(
        xs, tfmr_gmm_params, cs, tfmr_cs, ks*data_points_per_mode, ks)
    return avg_acc, avg_f1, avg_log_marginal

  compute_metrics = jax.jit(compute_metrics)

  def summarize(writer, step, params, key):
    k1, k2, k3 = jax.random.split(key, num=3)
    avg_acc, avg_f1, avg_log_marginal = compute_metrics(k1, params)
    writer.scalar("transformer/pairwise_acc", avg_acc, step=step)
    print("Transformer pairwise accuracy: %0.3f" % avg_acc)
    writer.scalar("transformer/pairwise_f1", avg_f1, step=step)
    print("Transformer pairwise f1: %0.3f" % avg_f1)
    writer.scalar("transformer/avg_log_marginal", avg_log_marginal, step=step)
    print("Transformer avg log marginal: %0.3f" % avg_log_marginal)

    if data_dim == 2:
      plot_params(min_k, min_k*data_points_per_mode, writer, step, params, k2)
      if FLAGS.plot_sklearn_comparison:
        plot_comparisons(writer, step, params)
    if step == 0:
      summarize_baselines(writer, step, k3)

  return summarize


def make_logdir(config):
  basedir = config.logdir
  exp_dir = (
      "nheads_%d"
      "_nenc_%d"
      "_ndec_%d"
      "_sepm_%0.2f"
      "_data_dim_%d"
      "_mink_%d"
      "_maxk_%d"
      "_dps_per_k_%d"
      "_cov_prior_%s"
      "_cov_dof_%d"
      "_%s"
      "_tpu%s" % (
        config.num_heads, config.num_encoders, config.num_decoders, 
        config.dist_multiplier, config.data_dim, config.min_k, config.max_k,
        config.data_points_per_mode, config.cov_prior, 
        config.cov_dof, config.normalization, config.tag)
      )
  return os.path.join(basedir, exp_dir)


def main(unused_argv):
  if FLAGS.cov_dof is None:
    FLAGS.cov_dof = FLAGS.data_dim + 2

  assert FLAGS.cov_dof >= FLAGS.data_dim + 2, "Wishart DOF must be >= 2 + data dim."

  if FLAGS.k is not None:
    FLAGS.min_k = FLAGS.k
    FLAGS.max_k = FLAGS.k

  if FLAGS.debug_nans:
    config.update("jax_debug_nans", True)

  if FLAGS.parallel and train.can_train_parallel():
    assert FLAGS.batch_size % jax.local_device_count(
    ) == 0, "Device count must evenly divide batch_size"
    FLAGS.batch_size = int(FLAGS.batch_size / jax.local_device_count())

  if FLAGS.plot_sklearn_comparison:
    assert FLAGS.min_k == 3 and FLAGS.max_k == 3

  FLAGS.dist_multiplier = oscipy.stats.chi2.ppf(FLAGS.dist_multiplier, df=FLAGS.data_dim)

  key = jax.random.PRNGKey(0)
  key, subkey = jax.random.split(key)
  model, init_params = make_model(
      key,
      model_name=FLAGS.model_name,
      num_encoders=FLAGS.num_encoders,
      num_decoders=FLAGS.num_decoders,
      num_heads=FLAGS.num_heads,
      value_dim=FLAGS.value_dim_per_head*FLAGS.num_heads,
      data_points_per_mode=FLAGS.data_points_per_mode,
      max_k=FLAGS.max_k,
      data_dim=FLAGS.data_dim)
  loss_fn = make_loss(
      model,
      model_name=FLAGS.model_name,
      min_k=FLAGS.min_k,
      max_k=FLAGS.max_k,
      data_points_per_mode=FLAGS.data_points_per_mode,
      cov_dof=FLAGS.cov_dof,
      cov_prior=FLAGS.cov_prior,
      dist_mult=FLAGS.dist_multiplier,
      mode_var=FLAGS.mode_var,
      data_dim=FLAGS.data_dim,
      batch_size=FLAGS.batch_size)
  summarize_fn = make_summarize(
      model,
      model_name=FLAGS.model_name,
      min_k=FLAGS.min_k,
      max_k=FLAGS.max_k,
      data_points_per_mode=FLAGS.data_points_per_mode,
      cov_dof=FLAGS.cov_dof,
      cov_prior=FLAGS.cov_prior,
      dist_mult=FLAGS.dist_multiplier,
      data_dim=FLAGS.data_dim,
      mode_var=FLAGS.mode_var,
      eval_batch_size=FLAGS.eval_batch_size)
  train.train_loop(
      subkey,
      init_params,
      loss_fn,
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
