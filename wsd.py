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
import numpy as np
import tensorflow as tf
from math import exp
from time import time
from data import BucketedBatchQueue

from wsd_utils import prepare_ptb_data
from wsd_utils import example_generator
from wsd_utils import space_tokenizer

from rnn_classifier import HParams
from rnn_classifier import RNNClassifier

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'mode', 'train', 'One of [train|eval|topk]')
flags.DEFINE_string(
  'corpus', 'ptb', 'Which dataset to train on')
flags.DEFINE_string(
  'data_dir', 'data', 'Data directory.')
flags.DEFINE_bool(
  'force_preprocess', False, 'Force preprocessing (overwrite existing)')
flags.DEFINE_string(
  'save_path', 'checkpoints', 'Directory for checkpoint files')
flags.DEFINE_integer(
  'save_secs', 60, 'Number of seconds between each model save.')
flags.DEFINE_integer(
  'eval_interval', 100, 'Number of training iterations')
flags.DEFINE_integer(
  'seed', 42, 'Seed for random number generator')
flags.DEFINE_integer(
  'num_training_iterations', 100000, 'Number of training iterations')
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

def run_epoch(sess, model, eval_op=None, verbose=False):
  tic = time()
  losses = 0.0
  iters = 0
  num_words = 0
  batch_size = FLAGS.batch_size
  epoch_size = FLAGS.batch_per_epoch
  fetches = {"loss": model.loss, "lens": model.lens}
  if eval_op is not None:
    fetches["eval_op"] = eval_op
  for step in range(epoch_size):
    vals = sess.run(fetches)
    loss = vals["loss"]
    lens = vals["lens"]
    total_len = sum(lens)
    losses += loss
    iters += batch_size
    num_words += total_len
    if verbose and step % (epoch_size // 10) == 10:
      infostr = "{:.3f} perplexity: {:.6f} speed {:.0f} wps".format(
        step * 1.0 / epoch_size,
        np.exp(losses / iters),
        iters * batch_size / (time() - tic))
      tf.logging.info(infostr)
  return np.exp(losses / iters)

def train(raw_train_path, raw_dev_path, model_config, batch_size,
          num_training_iterations, save_path='checkpoints', save_secs=60*5,
          report_interval=100):
  with tf.name_scope("Train"):
    train_queue = BucketedBatchQueue(raw_train_path, batch_size,
                                     is_training=True, name="train_queue",
                                     force_preprocess=FLAGS.force_preprocess)
    with tf.variable_scope("Model", reuse=None):
      m = RNNClassifier(model_config, train_queue, is_training=True)

  tf.summary.scalar("Training Loss", m.loss)
  tf.summary.scalar("Learning Rate", m.lr)

  with tf.name_scope("Valid"):
    valid_queue = BucketedBatchQueue(raw_dev_path, batch_size,
                                     is_training=False, name="valid_queue",
                                     force_preprocess=FLAGS.force_preprocess)
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

  saver = tf.train.Saver()

  hooks = [
    tf.train.CheckpointSaverHook(
      checkpoint_dir=save_path,
      save_secs=save_secs,
      saver=saver
    )
  ]

  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=save_path) as sess:
    start_iteration = sess.run(global_step)
    for train_iteration in range(start_iteration, num_training_iterations):
      if (train_iteration + 1) % report_interval == 0:
        train_loss_v, valid_loss_v, _ = sess.run(
          (m.loss, mvalid.loss, m.train_op))
        tf.logging.info('%d: Training loss %f. Validation loss %f.',
                        train_iteration,
                        train_loss_v,
                        valid_loss_v)
      else:
        train_loss_v, _ = sess.run((m.loss, m.train_op))

def get_model_config(vocab_path):
  vocab_size = 0
  with open(vocab_path) as f:
    for line in f:
      vocab_size += 1
  config = HParams(num_label=vocab_size, vocab_size=vocab_size,
                   embed_size=FLAGS.embed_size, hidden_size=FLAGS.hidden_size,
                   cell_type='gru', num_layer=2, keep_prob=1.0,
                   learning_rate=0.0001, grad_clip=10.0, optimizer='adam')
  return config

def main(_):
  random.seed(FLAGS.seed)
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.set_random_seed(FLAGS.seed)
  tf.logging.info('Using {} corpus'.format(FLAGS.corpus))
  if FLAGS.corpus == 'ptb':
    raw_train_path, raw_dev_path, vocab_path = prepare_ptb_data(
      FLAGS.data_dir,
      space_tokenizer,
      force=FLAGS.force_preprocess
    )
  else:
    raise ValueError('unrecognized corpus: {}'.format(FLAGS.corpus))
  config = get_model_config(vocab_path)
  train(raw_train_path, raw_dev_path, config, FLAGS.batch_size,
        FLAGS.num_training_iterations,
        save_path=FLAGS.save_path)

if __name__ == "__main__":
  tf.app.run()
