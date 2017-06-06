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
import itertools
from collections import namedtuple

import numpy as np
import tensorflow as tf

from tensorflow.contrib import rnn as contrib_rnn
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.contrib.rnn import GRUCell

def loss(logits,
         targets,
         average_across_batch=True,
         softmax_loss_function=None,
         name=None):
  with ops.name_scope(name, "cross_entropy", [logits, targets]):
    num_classes = array_ops.shape(logits)[-1]
    batch_size  = array_ops.shape(logits)[0]
    crossent    = None
    if softmax_loss_function is None:
      # A common use case is to have logits of shape
      # [batch_size, num_classes] and labels of shape
      # [batch_size]. But higher dimensions are supported.
      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
        labels=targets,
        logits=logits
      )
    else:
      crossent = softmax_loss_function(logits, targets)
    if average_across_batch:
      return tf.reduce_mean(crossent)
    return crossent

HParams = namedtuple('HParams',
                     'num_label, vocab_size, embed_size, hidden_size, '
                     'cell_type, num_layer, keep_prob, learning_rate,'
                     'grad_clip, optimizer')

class RNNClassifier(object):
  def __init__(self, config, input_data, is_training=True):
    self._cell_type = config.cell_type
    self._inputs, self._lens, self._targets = input_data.batch

    if config.num_label < 1 or config.vocab_size < 1:
      raise ValueError("must set num_label and vocab_size")

    self._init_embeddings(is_training, config)
    self._init_encoder(is_training, config)
    self._init_decoder(is_training, config)
    self._init_optimizer(is_training, config)

  def _init_embeddings(self, is_training, config):
    self._embedding = tf.get_variable("embedding", [config.vocab_size,
                                                    config.embed_size])
    self._input_embed = tf.nn.embedding_lookup(self._embedding, self._inputs,
                                               name="embedding_lookup")

  def _get_rnn_state(self, state):
    if self._cell_type == 'lstm':
      return state[-1].h
    else:
      return state[-1]

  def _init_encoder(self, is_training, config):
    assert self._input_embed != None, "call init_embeddings first"
    with tf.variable_scope("Encoder"):
      cells = []
      for l in range(config.num_layer):
        if self._cell_type == 'lstm':
          cell = core_rnn_cell_impl.BasicLSTMCell(config.hidden_size)
        elif self._cell_type == 'gru':
          cell = GRUCell(config.hidden_size)
        else:
          raise ValueError("unrecognized cell type: {}".format(self._cell_type))
        if is_training:
          cell = contrib_rnn.DropoutWrapper(cell, config.keep_prob)
        cells += [ cell ]

      self._encoder_cell = contrib_rnn.MultiRNNCell(cells)

      (fw_encoder_outputs, fw_encoder_state) = (
        tf.nn.dynamic_rnn(cell=self._encoder_cell,
                          inputs=self._input_embed,
                          time_major=False,
                          dtype=tf.float32))

      final_state = self._get_rnn_state(fw_encoder_state)
      final_state_size = config.hidden_size

      self._code_size = final_state_size
      self._codes = final_state

  def _init_decoder(self, is_training, config):
    assert self._codes != None, "call init_encoder first"

    with tf.variable_scope("Decoder"):
      # Softmax layer
      with tf.variable_scope("Softmax"):
        W = tf.get_variable('W', [self._code_size, config.num_label])
        b = tf.get_variable('b', [config.num_label],
                            initializer=tf.constant_initializer(0.0))

      # Get logits
      self._decoder_logits = tf.matmul(self._codes, W) + b
      self._decoder_predict = tf.argmax(self._decoder_logits,
                                        axis=-1,
                                        name='decoder_prediction_train')

      # Flatten while preserving the leading batch dim
      # targets = tf.contrib.layers.flatten(self._targets)
      targets = tf.reshape(self._targets, [-1])

      # Compute loss
      self._loss = loss(logits=self._decoder_logits,
                        targets=targets,
                        name="cross_entropy")

  def _init_optimizer(self, is_training, config):
    if not is_training:
      return
    assert self._decoder_logits != None, "call init_decoder first"

    self._lr = tf.get_variable(
      "learning_rate",
      shape=[],
      dtype=tf.float32,
      initializer=tf.constant_initializer(config.learning_rate),
      trainable=False
    )
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),
                                      config.grad_clip)

    optimizer = None
    if config.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self._lr)
    elif config.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(self._lr)
    else:
      raise ValueError('unsupported optimizer: {}'.format(config.optimizer))

    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def assign_lr(self, session, lr_value):
    session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def train_op(self):
    return self._train_op

  @property
  def lr(self):
    return self._lr

  @property
  def loss(self):
    return self._loss

  @property
  def encoder_inputs(self):
    return self._encoder_inputs

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def batch_per_epoch(self):
    return self._batch_per_epoch

  @property
  def lens(self):
    return self._lens

  @property
  def logits(self):
    return self._decoder_logits

  @property
  def guesses(self):
    return self._decoder_predict

  @property
  def code_size(self):
    return self._code_size

  @property
  def codes(self):
    return self._codes
