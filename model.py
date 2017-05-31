# Copyright 2017 Johns Hopkins University (Nicholas Andrews).
#
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

# pylint: disable=unused-import,g-bad-import-order
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# pylint: enable=unused-import

import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops
from tensorflow.contrib.rnn import GRUCell
from tensorflow.contrib.rnn import LSTMCell

def cross_entropy_loss(logits,
                       targets,
                       average_across_batch=True,
                       softmax_loss_function=None,
                       name=None):
  with ops.name_scope(name, "cross_entropy_loss", [logits, targets]):
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

class BasicWSD(object):
  def __init__(self, cfg, is_training=False):
    if cfg.num_label < 1 or cfg.vocab_size < 1:
      raise ValueError("must set num_label and vocab_size")
    self._init_placeholders(is_training, cfg)
    self._init_embeddings(is_training, cfg)
    self._init_encoder(is_training, cfg)
    self._init_decoder(is_training, cfg)
    self._init_optimizer(is_training, cfg)

  def _init_placeholders(self, is_training, cfg):
    self._input_placeholder = tf.placeholder(tf.int32, name="input_placeholder")
    self._target_placeholder = tf.placeholder(tf.int32,
                                              name="target_placeholder")

  def _init_embeddings(self, is_training, cfg):
    with tf.device("/cpu:0"):
      self._embedding = tf.get_variable(
        "embedding", [cfg.vocab_size, cfg.embed_size], dtype=cfg.dtype)
      self._input_embed = tf.nn.embedding_lookup(embedding,
                                                 self._input_placeholder)

  def _get_rnn_state(self, state):
    if self._cell_type == 'lstm':
      return state[-1].h
    else:
      return state[-1]

  def _init_encoder(self, is_training, cfg):
    assert self._input_embed != None, "call init_embeddings first"
    with tf.variable_scope("encoder"):
      cells = []
      for l in range(cfg.num_layer):
        if self._cell_type == 'lstm':
          cell = core_rnn_cell_impl.BasicLSTMCell(cfg.hidden_size)
        elif self._cell_type == 'gru':
          cell = GRUCell(cfg.hidden_size)
        else:
          raise ValueError("unrecognized cell type: {}".format(self._cell_type))
        if is_training:
          cell = contrib_rnn.DropoutWrapper(cell, cfg.keep_prob)
        cells += [ cell ]

      self._encoder_cell = contrib_rnn.MultiRNNCell(cells)

      (rnn_outputs, rnn_state) = (
        tf.nn.dynamic_rnn(cell=self._encoder_cell,
                          inputs=self._input_embed,
                          time_major=False,
                          dtype=cfg.dtype)
      )

      final_state = self._get_rnn_state(rnn_state)
      final_state_size = cfg.hidden_size
      self._code_size = final_state_size
      self._codes = final_state

  def _init_decoder(self, is_training, cfg):
    assert self._codes != None, "call init_encoder first"
    with tf.variable_scope("decoder"):
      with tf.variable_scope('softmax'):
        W = tf.get_variable('W', [self._code_size, cfg.num_label])
        b = tf.get_variable('b', [cfg.num_label],
                            initializer=tf.constant_initializer(0.0))
      self._decoder_logits = tf.matmul(self._codes, W) + b
      self._decoder_predict = tf.argmax(self._decoder_logits,
                                        axis=-1,
                                        name='decoder_prediction_train')
      self._loss = cross_entropy_loss(logits=self._decoder_logits,
                                      targets=self._targets)

  def _init_optimizer(self, is_training, cfg):
    if not is_training:
      return
    assert self._decoder_logits != None, "call init_decoder first"
    self._lr = tf.Variable(cfg.learning_rate, trainable=False,
                           name="learning_rate")
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self._loss, tvars),
                                      cfg.grad_clip)
    optimizer = None
    if cfg.optimizer == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(self._lr)
    elif cfg.optimizer == 'adam':
      optimizer = tf.train.AdamOptimizer(self._lr)
    else:
      raise ValueError('unsupported optimizer: {}'.format(cfg.optimizer))
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))
    self._new_lr = tf.placeholder(
      tf.float32, shape=[], name="new_learning_rate")
    self._lr_update = tf.assign(self._lr, self._new_lr)

  def step(self, sess, inp, target, eval_op=None, verbose=False):
    """Run a step of the network."""
    batch_size, length = inp.shape[0], inp.shape[1]
    train_mode = True
    fetches = {
      "cost": model.cost
    }
    if eval_op is not None:
      fetches["eval_op"] = eval_op
    feed_dict = {}
    feed_dict[self._input_placeholder.name] = inp
    feed_dict[self._target_placeholder.name] = target
    vals = sess.run(fetches, feed_dict)
    return vals

  def assign_lr(self, sess, lr_value):
    sess.run(self._lr_update, feed_dict={self._new_lr: lr_value})

  @property
  def global_step(self):
    return self._global_step

  @property
  def cur_length(self):
    return self._cur_length

  @property
  def train_op(self):
    return self._train_op

  @property
  def lr(self):
    return self._lr

  @property
  def lr_decay_op(self):
    return self._lr_decay_op
