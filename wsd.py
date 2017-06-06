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

from wsd_utils import prepare_ptb_data
from wsd_utils import initialize_vocabulary
from wsd_utils import ids_to_words
from wsd_utils import space_tokenizer

from data import BucketedBatchQueue
from data import InferenceBatchQueue

from rnn_classifier import HParams
from rnn_classifier import RNNClassifier

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'mode', 'train', 'One of [train|eval|topk]')
flags.DEFINE_string(
  'corpus', 'ptb', 'Which dataset to train on')
flags.DEFINE_string(
  'data_dir', 'data', 'Data directory to store preprocessed training data.')
flags.DEFINE_bool(
  'force_preprocess', False, 'Force preprocessing (overwrite existing)')
flags.DEFINE_integer(
  'K', 5, '[topk] Number of results to return in top K predictions.')
flags.DEFINE_string(
  'vocab_path', None, '[topk] Path to vocabulary file.')
flags.DEFINE_string(
  'input_path', None, '[topk] Path to input sentences for predictions.')
flags.DEFINE_string(
  'checkpoint_path', None, '[topk|eval] Path to model checkpoint to restore.')
flags.DEFINE_integer(
  'num_eval_batch', 250, ['[eval] Number of evaluation batches to use.'])
flags.DEFINE_string(
  'save_path', 'checkpoints', 'Directory for checkpoint files')
flags.DEFINE_integer(
  'save_secs', 60*5, 'Number of seconds between each model save.')
flags.DEFINE_integer(
  'report_interval', 1000, 'Number of iterations between status updates.')
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
  'hidden_size', 512, 'RNN state size')
flags.DEFINE_integer(
  'embed_size', 256, 'Input symbol embedding size')
flags.DEFINE_bool(
  'lowercase', True, 'Lower case all words.')
flags.DEFINE_float(
  'keep_prob', 0.75, 'Dropout probability.')
flags.DEFINE_integer(
  'num_layer', 3, 'Number of stacked RNN layers.')
flags.DEFINE_float(
  'learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float(
  'grad_clip', 10.0, 'Gradient norm clip.')

def eval(raw_data_path, config, batch_size):
  if not FLAGS.checkpoint_path:
    raise ValueError('must provide checkpoint path')

  with tf.name_scope("Valid"):
    q = BucketedBatchQueue(raw_data_path, batch_size,
                           is_training=False, name="valid_queue",
                           force_preprocess=FLAGS.force_preprocess)
    with tf.variable_scope("Model"):
      m = RNNClassifier(config, q.batch, is_training=False)

  saver = tf.train.Saver()
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  with tf.Session() as sess:
    # Initialize queues
    sess.run(init_op)

    # Restore model parameters
    tf.logging.info('Restoring model: {}'.format(FLAGS.checkpoint_path))
    saver.restore(sess, FLAGS.checkpoint_path)

    # Run inference & print top K to stdout
    tf.logging.info('Computing held-out perplexity...')
    total_loss = 0
    nbatch = 0
    with tf.contrib.slim.queues.QueueRunners(sess):
      for _ in range(FLAGS.num_eval_batch):
        loss_v = sess.run(m.loss)
        total_loss += loss_v
        nbatch += 1
    return total_loss / nbatch

def train(raw_train_path, raw_dev_path, model_config, batch_size,
          num_training_iterations, save_path='checkpoints', save_secs=60,
          report_interval=1000):
  with tf.name_scope("Train"):
    train_queue = BucketedBatchQueue(raw_train_path, batch_size,
                                     is_training=True, name="train_queue",
                                     force_preprocess=FLAGS.force_preprocess)
    with tf.variable_scope("Model", reuse=None):
      m = RNNClassifier(model_config, train_queue.batch, is_training=True)

  with tf.name_scope("Valid"):
    valid_queue = BucketedBatchQueue(raw_dev_path, batch_size,
                                     is_training=False, name="valid_queue",
                                     force_preprocess=FLAGS.force_preprocess)
    with tf.variable_scope("Model", reuse=True):
      mvalid = RNNClassifier(model_config, valid_queue.batch, is_training=False)

  saver = tf.train.Saver(max_to_keep=5,
                         keep_checkpoint_every_n_hours=1)

  hooks = [
    tf.train.CheckpointSaverHook(
      checkpoint_dir=save_path,
      save_secs=save_secs,
      saver=saver
    )
  ]

  tf.logging.info('Report interval: %d', report_interval)
  with tf.train.SingularMonitoredSession(
      hooks=hooks, checkpoint_dir=save_path) as sess:
    start_iteration = sess.run(m.global_step)
    tf.logging.info('Start iteration: %d of %d', start_iteration,
                    num_training_iterations)
    for train_iteration in range(start_iteration, num_training_iterations):
      if (train_iteration + 1) % report_interval == 0:
        train_loss_v, valid_loss_v, _ = sess.run(
          (m.loss, mvalid.loss, m.train_op))
        tf.logging.info('%d of %d: Training loss %f. Validation loss %f. Global step %d',
                        train_iteration,
                        num_training_iterations,
                        train_loss_v,
                        valid_loss_v,
                        sess.run(m.global_step))
      else:
        train_loss_v, _ = sess.run((m.loss, m.train_op))

def get_model_config(vocab_size):
  config = HParams(num_label=vocab_size, vocab_size=vocab_size,
                   embed_size=FLAGS.embed_size, hidden_size=FLAGS.hidden_size,
                   cell_type='gru', num_layer=FLAGS.num_layer,
                   keep_prob=FLAGS.keep_prob, K=FLAGS.K,
                   learning_rate=FLAGS.learning_rate, grad_clip=FLAGS.grad_clip,
                   optimizer='adam')
  return config

def run_training():
  random.seed(FLAGS.seed)
  tf.set_random_seed(FLAGS.seed)
  tf.logging.info('Using {} corpus'.format(FLAGS.corpus))
  if FLAGS.corpus == 'ptb':
    raw_train_path, raw_dev_path, vocab_path = prepare_ptb_data(
      FLAGS.data_dir,
      space_tokenizer,
      lowercase=FLAGS.lowercase,
      force=FLAGS.force_preprocess
    )
  else:
    raise ValueError('unrecognized corpus: {}'.format(FLAGS.corpus))

  # Read vocabulary
  vocab, rev_vocab = initialize_vocabulary(vocab_path)
  vocab_size = len(vocab)

  # Get model configuration
  config = get_model_config(vocab_size)

  # Train model (maybe resume existing checkpoint)
  train(raw_train_path, raw_dev_path, config, FLAGS.batch_size,
        FLAGS.num_training_iterations, save_path=FLAGS.save_path,
        save_secs=FLAGS.save_secs, report_interval=FLAGS.report_interval)

def run_eval():
  tf.logging.info('Using {} corpus'.format(FLAGS.corpus))
  if FLAGS.corpus == 'ptb':
    raw_train_path, raw_dev_path, vocab_path = prepare_ptb_data(
      FLAGS.data_dir,
      space_tokenizer,
      lowercase=FLAGS.lowercase,
      force=FLAGS.force_preprocess
    )
  else:
    raise ValueError('unrecognized corpus: {}'.format(FLAGS.corpus))

  # Read vocabulary
  vocab, rev_vocab = initialize_vocabulary(vocab_path)
  vocab_size = len(vocab)

  # Get model configuration
  config = get_model_config(vocab_size)

  # Compute perplexity
  mean_perplexity = eval(raw_dev_path, config, FLAGS.batch_size)

  tf.logging.info('Mean held-out perplexity: %f', mean_perplexity)

def run_topk():
  if not FLAGS.vocab_path:
    raise ValueError('must supply path to vocabulary file: --vocab_path')
  if not FLAGS.input_path:
    raise ValueError('must supply input sentences: --input_path')
  if not FLAGS.checkpoint_path:
    raise ValueError('must supply path to model checkpoint: --checkpoint_path')

  # Read vocabulary
  vocab, rev_vocab = initialize_vocabulary(FLAGS.vocab_path)

  # Prepare input queue
  q = InferenceBatchQueue(FLAGS.input_path, vocab, FLAGS.batch_size,
                          lowercase=FLAGS.lowercase)

  # Get model configuration
  config = get_model_config(len(vocab))

  # Create inference graph
  with tf.variable_scope("Model"):
    m = RNNClassifier(config, q.batch, is_training=False)

  saver = tf.train.Saver()
  init_op = tf.group(tf.global_variables_initializer(),
                     tf.local_variables_initializer())
  with tf.Session() as sess:
    # Initialize queues
    sess.run(init_op)

    # Restore model parameters
    tf.logging.info('Restoring model: {}'.format(FLAGS.checkpoint_path))
    saver.restore(sess, FLAGS.checkpoint_path)

    # Run inference & print top K to stdout
    with tf.contrib.slim.queues.QueueRunners(sess):
      topk_v = sess.run([m.topk_ids])
      print(type(topk_v))
      for tk in ids_to_words(topk_v, rev_vocab):
        print(tk)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  if FLAGS.mode == 'train':
    run_training()
  elif FLAGS.mode == 'eval':
    run_eval()
  elif FLAGS.mode == 'topk':
    run_topk()
  else:
    raise ValueError('unrecognized mode: {}'.format(FLAGS.mode))

if __name__ == "__main__":
  tf.app.run()
