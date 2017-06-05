#! /usr/bin/env python3

# Copyright 2017 Johns Hopkins University (Nicholas Andrews).
# All Rights Reserved.
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
# ==============================================================================

import math
import sys
import os
import pickle
import random
import itertools
from timeit import default_timer as timer
import numpy as np
import tensorflow as tf
from math import exp
from tqdm import tqdm

from data import instances_to_tensors
from data import prepare
from data import BatchQueue

from rnn_classifier import RNNClassifierConfig
from rnn_classifier import RNNClassifier

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'train_path', None, 'Path to training instances')
flags.DEFINE_string(
  'valid_path', None, 'Path to validation instances')
flags.DEFINE_string(
  'save_path', None, 'Directory for checkpoint files')
flags.DEFINE_integer(
  'seed', 42, 'Seed for random number generator')
flags.DEFINE_integer(
  'max_epoch', 100, 'Seed for random number generator')
flags.DEFINE_integer(
  'batch_size', 32, 'Batch size for gradient computation')
flags.DEFINE_integer(
  'max_train', None, 'Maximum number of training instances')
flags.DEFINE_integer(
  'max_valid', None, 'Maximum number of validation instances')
flags.DEFINE_integer(
  'hidden_size', 256, 'RNN state size')
flags.DEFINE_integer(
  'embed_size', 128, 'Input symbol embedding size')

def get_input_queue(batch_size, examples, is_training, name=None):
  return BatchQueue(examples, batch_size, is_training, name=name)

def run_epoch(sess, model, eval_op=None, verbose=False):
  tic = timer()
  losses = 0.0
  iters = 0
  num_words = 0
  batch_size = model.batch_size
  epoch_size = model.batch_per_epoch
  fetches = {"loss": model.loss}
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  for step in range(epoch_size):
    vals = sess.run(fetches)
    loss = vals["loss"]
    losses += loss
    iters += batch_size
    num_words += model.total_len
    if verbose and step % (epoch_size // 10) == 10:
      infostr = "{:.3f} perplexity: {:.6f} speed {:.0f} wps".format(
        step * 1.0 / epoch_size,
        np.exp(losses / iters),
        iters * batch_size / (timer() - tic))
      tf.logging.info(infostr)
  return np.exp(losses / iters)

def train(train_examples, valid_examples, model_config, max_epoch,
          batch_size, save_path = None):
  with tf.name_scope("Train"):
    train_queue = get_input_queue(batch_size, train_examples, name="TrainInput")
    with tf.variable_scope("Model", reuse=None):
      m = RNNClassifier(model_config, train_queue, is_training=True)

  tf.summary.scalar("Training Loss", m.loss)
  tf.summary.scalar("Learning Rate", m.lr)

  with tf.name_scope("Valid"):
    valid_queue = get_input_queue(batch_size, valid_examples, name="ValidInput")
    with tf.variable_scope("Model", reuse=True):
      mvalid = RNNClassifier(model_config, valid_queue, is_training=False)

  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())

  global_step = tf.get_variable(
    name="global_step",
    shape=[],
    dtype=tf.int64,
    initializer=tf.zeros_initializer(),
    trainable=False,
    collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

  tf.logging.info("Starting session.")
  with tf.Session() as sess:
    tf.logging.info("Initializing global and local variables.")
    sess.run(init_op)

    tf.logging.info("Moving training data to device.")
    train_queue.init(train_data, sess)
    tf.logging.info("Moving validation data to device.")
    valid_queue.init(valid_data, sess)

    tf.train.start_queue_runners(sess=sess)
    for i in range(max_epoch):
      m.assign_lr(sess, model_config.learning_rate)
      tf.logging.info("Epoch: {} Learning rate: {:.6f}".format(
        i + 1, sess.run(m.lr)))
      train_perplexity = run_epoch(sess, m, eval_op=m.train_op,
                                   verbose=True)
      tf.logging.info("Epoch: {} Train perplexity: {:.6f}".format(
        i + 1, train_perplexity))

      valid_perplexity = run_epoch(sess, mvalid)
      tf.logging.info("Epoch: {} Valid perplexity: {:.6f}".format(
        i + 1, valid_perplexity))

    if save_path:
      tf.logging.info("Saving model to: {}".format(save_path))
      sv.saver(sess, save_path, global_step=global_step)

def get_model_config(num_tags, num_syms):
  config = RNNClassifierConfig(num_label=num_tags, vocab_size=num_syms,
                               embed_size=FLAGS.embed_size,
                               hidden_size=FLAGS.hidden_size,
                               cell_type='gru', num_layer=2, keep_prob=1.0,
                               learning_rate=0.0001, grad_clip=10.0,
                               optimizer='adam')
  return config

def main(_):
  random.seed(FLAGS.seed)
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(FLAGS.seed)
  tf.logging.info('Loading training data: {}'.format(FLAGS.train_path))
  tf.logging.info('Loading validation data: {}'.format(FLAGS.valid_path))
  corpus = prepare(FLAGS.train_path,
                   FLAGS.valid_path,
                   max_train_examples=FLAGS.max_train,
                   max_valid_examples=FLAGS.max_valid)
  train_examples, valid_examples, syms, tags = corpus
  config = get_model_config(len(tags), len(syms))
  train(train_examples, valid_examples, config,
        FLAGS.max_epoch, FLAGS.batch_size, save_path=FLAGS.save_path)

if __name__ == "__main__":
  tf.app.run()
