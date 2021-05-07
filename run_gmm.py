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
                  ["mean_scale_weight", "mean", "msw_unconditional", "mean_unconditional", "fixed_k"],
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
                     "the maximum number of modes in the data.")
flags.DEFINE_enum("dist", "l2", ["l2", "kl", "symm_kl"],
                  "The distance function used to measure similarity of components in the loss.")
flags.DEFINE_integer("data_points_per_mode", 25,
                     "Number of data points to include per mode in the data.")
flags.DEFINE_string("test_data_points_per_mode", None,
                     "Number of data points per mode in the test data, defaults to"
                     " data_points_per_mode.")
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
flags.DEFINE_integer("expensive_summarize_every", 10000,
                     "Number of steps between expensive summaries.")
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

sampling_types = {
 "mean": "mean",
 "mean_unconditional": "mean",
 "mean_scale": "mean_scale",
 "mean_scale_weight": "mean_scale_weight",
 "msw_unconditional": "mean_scale_weight",
 "fixed_k": "mean_scale_weight"
}

class_dict = {
      "mean": gmm_models.OriginalMeanInferenceMachine,
      #"mean_scale": gmm_models.MeanScaleInferenceMachine,
      "mean_scale_weight": gmm_models.MSWOriginal,
      "msw_unconditional": gmm_models.MSWUnconditional,
      "mean_unconditional": gmm_models.UnconditionalMeanInferenceMachine,
      "fixed_k": gmm_models.UnconditionalFixedK,
}

def make_model(key,
               model_name="mean",
               num_encoders=4,
               num_decoders=4,
               num_heads=8,
               value_dim=128,
               data_points_per_mode=25,
               max_k=10,
               data_dim=2,
               normalization="no_norm",
               dist="l2"):
  model = class_dict[model_name](
      data_dim=data_dim, max_k=max_k, algo_k=max_k, num_heads=num_heads,
      num_encoders=num_encoders, num_decoders=num_decoders, qkv_dim=value_dim,
      normalization=normalization, dist=dist)
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
        key, sampling_types[model_name], batch_size, min_k, max_k, max_k*data_points_per_mode,
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
    data_points_per_mode=50,
    test_data_points_per_mode=[25],
    cov_dof=10,
    cov_prior="inv_wishart",
    dist_mult=2.,
    data_dim=2,
    mode_var=1.,
    eval_batch_size=256):

  def sample_eval_batch(key, points_per_mode):
    xs, cs, ks, params = sample_gmm.sample_batch_random_ks(
        key, sampling_types[model_name], eval_batch_size, min_k, max_k, 
        2 * max_k * points_per_mode, data_dim, mode_var, cov_dof, cov_prior, dist_mult)
    train_xs = xs[:, :max_k * points_per_mode]
    test_xs = xs[:, max_k * points_per_mode:]
    train_cs = cs[:, :max_k * points_per_mode]
    test_cs = cs[:, max_k * points_per_mode:]
    return train_xs, test_xs, train_cs, test_cs, ks, params

  sample_eval_batch = jax.jit(sample_eval_batch, static_argnums=(1,))

  def sample_single_gmm(key, points_per_mode, num_modes):
    xs, cs, params = sample_gmm.sample_batch_fixed_ks(
        key, sampling_types[model_name], jnp.array([num_modes]), max_k, 
        max_k * points_per_mode, data_dim, mode_var, cov_dof, cov_prior, dist_mult)

    return xs[0], cs[0], (params[0][0], params[1][0], params[2][0])

  def model_classify(params, inputs, ks, points_per_mode):
    return gmm_models.classify_with_defaults(
        model, params, inputs, eval_batch_size, ks*points_per_mode, ks,
        max_k, jnp.eye(data_dim)*mode_var)

  def sample_and_classify_eval_batch(key, params, points_per_mode):
    train_xs, test_xs, train_cs, test_cs, ks, true_gmm_params = sample_eval_batch(key,
        points_per_mode)
    tfmr_train_cs, tfmr_gmm_params = model_classify(params, train_xs, ks, points_per_mode)
    tfmr_test_cs = jax.vmap(gmm_models.masked_classify_points)(
            test_xs, tfmr_gmm_params[0], tfmr_gmm_params[1], tfmr_gmm_params[2], ks)
    return (train_xs, test_xs, tfmr_train_cs, train_cs, tfmr_test_cs, test_cs, ks, 
            true_gmm_params, tfmr_gmm_params)

  def sample_and_classify_single_gmm(key, params, points_per_mode, num_modes):
    xs, cs, gmm_params = sample_single_gmm(key, points_per_mode, num_modes)
    tfmr_cs, tfmr_gmm_params = model_classify(
        params, xs[jnp.newaxis], jnp.array([num_modes]), points_per_mode)
    return xs, cs, gmm_params, tfmr_cs, tfmr_gmm_params

  sample_and_classify_single_gmm = jax.jit(sample_and_classify_single_gmm, static_argnums=(2,))
  sample_and_classify_eval_batch = jax.jit(sample_and_classify_eval_batch, static_argnums=(2,))

  def write_metrics(writer, step, key, metrics):
    for name, metric in metrics.items():
      writer.scalar("%s/%s" % (key, name), metric, step=step)
      print("%s %s: %0.3f" % (key, name, metric))

  def summarize_baselines(writer, step, key):
    key, subkey = jax.random.split(key)
    train_xs, test_xs, train_cs, test_cs, ks, _ = sample_eval_batch(subkey, data_points_per_mode)
    em_metrics = gmm_eval.compute_masked_baseline_metrics(
          train_xs, train_cs, test_xs, test_cs, sampling_types[model_name], mode_var, 
          ks, ks*data_points_per_mode)
    write_metrics(writer, step, "EM train",
        dict(zip(["pairwise acc", "pairwise f1", "ll"], em_metrics[:3])))
    write_metrics(writer, step, "EM test",
        dict(zip(["pairwise acc", "pairwise f1", "ll"], em_metrics[3:])))
  
  def compute_metrics(key, params, points_per_mode):
    (train_xs, test_xs, 
     tfmr_train_cs, train_cs, 
     tfmr_test_cs, test_cs, 
     ks, true_gmm_params, tfmr_gmm_params) = sample_and_classify_eval_batch(key, params,
         points_per_mode)
    train_acc, train_f1, train_log_marginal = gmm_eval.compute_metrics(
        train_xs, tfmr_gmm_params, train_cs, tfmr_train_cs, ks*points_per_mode, ks)
    test_acc, test_f1, test_log_marginal = gmm_eval.compute_metrics(
        test_xs, tfmr_gmm_params, test_cs, tfmr_test_cs, ks*points_per_mode, ks)
    return train_acc, train_f1, train_log_marginal, test_acc, test_f1, test_log_marginal

  compute_metrics = jax.jit(compute_metrics, static_argnums=(2,))

  def summarize(writer, step, params, key):
    k1, k2 = jax.random.split(key)
    (train_acc, train_f1, train_ll, 
        test_acc, test_f1, test_ll) = compute_metrics(k1, params, data_points_per_mode)
    write_metrics(writer, step, "Transformer train",
        {"pairwise acc": train_acc, "pairwise f1": train_f1, "ll": train_ll})
    write_metrics(writer, step, "Transformer test",
        {"pairwise acc": test_acc, "pairwise f1": test_f1, "ll": test_ll})
    if step == 0:
      summarize_baselines(writer, step, k2)

  def plot_params(num_modes, num_data_points, writer, step, params, key):
    outs = sample_and_classify_single_gmm(key, params, num_data_points // num_modes, num_modes)
    xs, true_cs, true_params, pred_cs, pred_params = outs
    pred_cs = pred_cs[0]
    pred_params = (pred_params[0][0], pred_params[1][0], pred_params[2][0])
    em_cs, em_params = gmm_eval.em_fit_and_predict(
        xs, num_modes, sampling_types[model_name], mode_var)
    fig = plotting.plot_em_comparison(
      xs, num_modes, true_cs, true_params, pred_cs, pred_params, em_cs,
      em_params)
    plot_img = plotting.plot_to_numpy_image(plt)
    writer.image(
        "%d_modes_%d_points" % (num_modes, num_data_points),
        plot_img, step=step)
    plt.close(fig)

  def eval_on_different_sized_datasets(writer, step, params, key):
    for ppm in test_data_points_per_mode:
      key, k1 = jax.random.split(key)
      (train_acc, train_f1, train_ll, 
          test_acc, test_f1, test_ll) = compute_metrics(k1, params, ppm)
      write_metrics(writer, step, "Transformer train diff dataset sizes",
          {"pairwise acc %d" % ppm: train_acc, "pairwise f1 %d" % ppm: train_f1, "ll %d" % ppm: train_ll})
      write_metrics(writer, step, "Transformer test diff dataset sizes",
          {"pairwise acc %d" % ppm: test_acc, "pairwise f1 %d" % ppm: test_f1, "ll %d" % ppm: test_ll})

  def expensive_summarize(writer, step, params, key):
    if data_dim == 2:
      for k in range(min_k, max_k+1):
        plot_params(k, k*data_points_per_mode, writer, step, params, key)
    if test_data_points_per_mode is not None:
      eval_on_different_sized_datasets(writer, step, params, key)
  
  return summarize, expensive_summarize


def make_logdir(config):
  basedir = config.logdir
  exp_dir = (
      "%s"
      "_nheads_%d"
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
      "_dist_%s"
      "_lr_%0.3f"
      "_tpu%s" % (
        config.model_name,
        config.num_heads, config.num_encoders, config.num_decoders, 
        config.dist_multiplier, config.data_dim, config.min_k, config.max_k, 
        config.data_points_per_mode, config.cov_prior, 
        config.cov_dof, config.normalization, config.dist, config.lr, config.tag)
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

  if FLAGS.test_data_points_per_mode is not None:
    FLAGS.test_data_points_per_mode = [int(x) for x in FLAGS.test_data_points_per_mode.split(",")]

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
      data_dim=FLAGS.data_dim, 
      normalization=FLAGS.normalization,
      dist=FLAGS.dist)
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
  summarize_fn, expensive_summarize_fn = make_summarize(
      model,
      model_name=FLAGS.model_name,
      min_k=FLAGS.min_k,
      max_k=FLAGS.max_k,
      data_points_per_mode=FLAGS.data_points_per_mode,
      test_data_points_per_mode=FLAGS.test_data_points_per_mode,
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
      expensive_summarize_fn=expensive_summarize_fn,
      summarize_every=FLAGS.summarize_every,
      expensive_summarize_every=FLAGS.expensive_summarize_every,
      checkpoint_every=FLAGS.checkpoint_every,
      clobber_checkpoint=FLAGS.clobber_checkpoint,
      logdir=make_logdir(FLAGS))
if __name__ == "__main__":
  app.run(main)
