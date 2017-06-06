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
from wsd_utils import sentence_to_token_ids
from wsd_utils import instances_from_ids
from wsd_utils import example_generator
from wsd_utils import space_tokenizer

from data import BucketedBatchQueue

from rnn_classifier import HParams
from rnn_classifier import RNNClassifier

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
  'mode', 'train', 'One of [train|topk]')
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
  'checkpoint_path', None, '[topk] Path to model checkpoint to restore.')
flags.DEFINE_string(
  'save_path', 'checkpoints', 'Directory for checkpoint files')
flags.DEFINE_integer(
  'save_secs', 60, 'Number of seconds between each model save.')
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
  'hidden_size', 256, 'RNN state size')
flags.DEFINE_integer(
  'embed_size', 128, 'Input symbol embedding size')

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

  saver = tf.train.Saver(max_to_keep=10,
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

def get_model_config(vocab_path):
  vocab_size = 0
  with open(vocab_path) as f:
    for line in f:
      vocab_size += 1
  config = HParams(num_label=vocab_size, vocab_size=vocab_size,
                   embed_size=FLAGS.embed_size, hidden_size=FLAGS.hidden_size,
                   cell_type='gru', num_layer=2, keep_prob=1.0, K=FLAGS.K,
                   learning_rate=0.0001, grad_clip=10.0, optimizer='adam')
  return config

def run_training():
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
        FLAGS.num_training_iterations, save_path=FLAGS.save_path,
        save_secs=FLAGS.save_secs, report_interval=FLAGS.report_interval)

def run_topk():
  if not FLAGS.vocab_path:
    raise ValueError('must supply path to vocabulary file: --vocab_path')
  if not FLAGS.input_path:
    raise ValueError('must supply input sentences: --input_path')
  if not FLAGS.checkpoint_path:
    raise ValueError('must supply path to model checkpoint: --checkpoint_path')

  # Get model configuration
  config = get_model_config(FLAGS.vocab_path)

  # Prepare inputs
  vocab, _ = initialize_vocabulary(FLAGS.vocab_path)
  tf.logging.info("{} entries in the vocabulary".format(len(vocab)))
  seqs = []
  lens = []
  targets = []
  with tf.gfile.GFile(FLAGS.input_path) as data_file:
    for line in data_file:
      token_ids = sentence_to_token_ids(line, vocab, space_tokenizer)
      max_len = 0
      for context, target in instances_from_ids(token_ids):
        seqs.append(context)
        l = len(context)
        if l > max_len:
          max_len = l
        lens.append(l)
        targets.append(target)
  nexample = len(seqs)
  seq_array = np.zeros([nexample, max_len], dtype=np.int64)
  len_array = np.zeros([nexample], dtype=np.int64)
  target_array = np.zeros([nexample], dtype=np.int64)
  for i in range(nexample):
    seq = seqs[i]
    for j in range(len(seq)):
      seq_array[i][j]
    len_array[i] = lens[i]
    target_array[i] = targets[i]
  inputs = (seq_array, len_array, target_array)
  inputs = tf.train.slice_input_producer(inputs,
                                         num_epochs=1,
                                         shuffle=False,
                                         capacity=FLAGS.batch_size * 2)
  batch = tf.train.batch(inputs, FLAGS.batch_size,
                         allow_smaller_final_batch=True)

  # Prepare model
  m = RNNClassifier(config, batch, is_training=False)

  # Run inference
  saver = tf.train.Saver()
  with tf.Session() as sess:
    tf.logging.info('Restoring model: {}'.format(FLAGS.checkpoint_path))
    saver.restore(sess, FLAGS.checkpoint_path)
    with tf.contrib.slim.queues.QueueRunners(sess):
      topk_v = sess.run(m.topk)
      print(topk_v)

def main(_):
  if FLAGS.mode == 'train':
    run_training()
  elif FLAGS.mode == 'topk':
    run_topk()
  else:
    raise ValueError('unrecognized mode: {}'.format(FLAGS.mode))

if __name__ == "__main__":
  tf.app.run()
