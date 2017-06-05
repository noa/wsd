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

import random
import os
import math
import collections
import numpy as np
import tensorflow as tf

from record_io import write_records

from data_reader import examples_queue
from data_reader import batch_examples

DataSet = collections.namedtuple('Data', 'sequences labels')

def _enc(s,d):
  if not (s in d):
    d[s] = len(d)
  return d[s]

def _encode_line(line, syms, tags, eos=None, sep="\t", space_str=None,
                 split_by_words=False):
  parts = line.rstrip().split(sep)
  if len(parts) != 2:
    raise ValueError('num parts {}'.format(len(parts)))
  raw_label = parts[0]
  enc_label = _enc(raw_label, tags)
  raw_seq = parts[1].split()
  if split_by_words:
    ret = []
    for word in raw_seq:
      ret.append([_enc(x, syms) for x in word])
    return (enc_label, ret)
  else:
    ret = []
    for i in range(len(raw_seq)):
      word = raw_seq[i]
      for char in word:
        ret.append(_enc(char, syms))
      if i+1 < len(raw_seq) and space_str:
        ret.append(_enc(space_str, syms))
    if eos:
      ret.append(_enc(eos, syms))
    return (enc_label, ret)

# return: [(label, text), ...]
def ingest(path, syms, tags, eos=None, max_examples=0, space_str=None,
           split_by_words=False):
  instances = []
  for line in open(path):
    instances.append(_encode_line(line, syms, tags, eos=eos,
                                  space_str=space_str,
                                  split_by_words=split_by_words))
    if max_examples > 0 and len(instances) == max_examples:
      break
  random.shuffle(instances)
  return instances

def get_random_batch(instances, batch_size):
  inputs = []
  targets = []
  for _ in range(batch_size):
    instance = random.choice(instances)
    assert(len(instance) == 2)
    inputs.append(instance[1])
    targets.append(instance[0])
  return (inputs, targets)

def get_next_batch(instances, batch_size, pos=0):
  inputs = []
  targets = []
  n = len(instances)
  if pos == n:
    return None
  for _ in range(batch_size):
    instance = instances[pos]
    assert(len(instance) == 2)
    inputs.append(instance[1])
    targets.append(instance[0])
    pos += 1
    if pos == n:
      break
  return ((inputs, targets), pos)

def format_input_seq(seq, symtab):
  toks = []
  for x in seq:
    if x > 0:
      toks.append(symtab[x])
  return "".join(toks)

def prepare(train_path, test_path, pad='<pad>', eos='<eos>',
            max_train_examples=0, max_valid_examples=0, space_str='<space>'):
  syms = {pad: 0}
  tags = {}
  train = ingest(train_path, syms, tags, eos=eos,
                 max_examples=max_train_examples, space_str=space_str)
  ntotal = len(train)
  valid = ingest(test_path, syms, tags, eos=eos,
                 max_examples=max_valid_examples, space_str=space_str)
  ntotal = len(valid)
  return train, valid, syms, tags

def instances_to_tensors(instances, dtype=np.int32):
  lens = [len(x[1]) for x in instances]
  M = max(lens)
  N = len(instances)
  inputs = np.zeros([N, M], dtype=dtype)
  lens = np.zeros([N], dtype=dtype)
  labels = np.zeros([N], dtype=dtype)
  for i in range(len(lens)):
    seq = instances[i][1]
    lens[i] = len(seq)
    labels[i] = instances[i][0]
    for j in range(len(seq)):
      inputs[i][j] = seq[j]
  return (inputs, lens, labels)

class BatchQueue(object):
  def __init__(self, examples, batch_size, is_training, name=None):
    record_path = path + '.tfrecords'
    tf.logging.info('Writing TF records to: {}'.format(record_path))
    write_records(examples, record_path)
    self._record_path = record_path
    queue = examples_queue(
      data_sources=record_path,
      data_fields_to_features={
        'sequence': tf.VarLenFeature(tf.int64),
        'label': tf.FixedLenFeature(tf.int64)
      },
      training=is_training
    )
    self._batch = batch_examples(queue, batch_size)

  def init(self):
    return

  @property
  def batch(self):
    return self._batch

InMemoryBatchQueueConfig = collections.namedtuple('InMemoryBatchQueueConfig',
                                                  'batch_size num_example max_len')

class InMemoryBatchQueue(object):
  def __init__(self, config, name=None):
    with tf.name_scope(name, "InMemoryBatchQueue"):
      self._seq_init = tf.placeholder(dtype=tf.int32,
                                      shape=(config.num_example,
                                             config.max_len))
      self._len_init = tf.placeholder(dtype=tf.int32,
                                      shape=(config.num_example))
      self._label_init = tf.placeholder(dtype=tf.int32,
                                        shape=(config.num_example))
      self._input_seqs = tf.Variable(self._seq_init, trainable=False,
                                     collections=[], name="input_seqs")
      self._input_lens = tf.Variable(self._len_init, trainable=False,
                                     collections=[], name="input_lens")
      self._input_targets = tf.Variable(self._label_init, trainable=False,
                                        collections=[], name="input_targets")
      self._batch_size = config.batch_size
      self._batch_per_epoch = math.ceil(config.num_example / config.batch_size)
      tensor_list = [self._input_seqs, self._input_lens, self._input_targets]
      sliced = tf.train.slice_input_producer(tensor_list, shuffle=True)
      self._slice = sliced
      batch = tf.train.batch(tensors=sliced,
                             batch_size=config.batch_size,
                             dynamic_pad=False,
                             enqueue_many=False)
      self._batch = batch

  def init(self, inputs, sess):
    sess.run(self._input_seqs.initializer,
             feed_dict={self._seq_init: inputs[0]})
    sess.run(self._input_lens.initializer,
             feed_dict={self._len_init: inputs[1]})
    sess.run(self._input_targets.initializer,
             feed_dict={self._label_init: inputs[2]})

  @property
  def batch(self):
    return self._batch

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def batch_per_epoch(self):
    return self._batch_per_epoch

class DataTest(tf.test.TestCase):
  def setUp(self):
    self._train_data = "\n".join(
      ["1\taaaa",
       "2\tbbbbbbb",
       "3\tcccccccccc",
       "4\tddddddddddddddd"])
    self._test_data = "\n".join(
      ["1\tone two three",
       "2\three two one"])

  def write_data(self):
    tmpdir = tf.test.get_temp_dir()
    testpath = os.path.join(tmpdir, 'test')
    trainpath = os.path.join(tmpdir, 'train')
    with open(trainpath, 'w') as f:
      f.write(self._train_data)
    with open(testpath, 'w') as f:
      f.write(self._test_data)
    return (trainpath, testpath)

  def testIngester(self):
    trainpath, testpath = self.write_data()
    eos='<eos>'
    train, valid, syms, tags = prepare(trainpath, testpath, eos=eos)
    rev_syms = {v: k for k, v in syms.items()}
    assert len(train) == len(self._train_data.split('\n'))
    assert len(valid) == len(self._test_data.split('\n'))
    for i in range(2):
      assert train[i][1][-1] == syms[eos]

  def testToTensor(self):
    trainpath, testpath = self.write_data()
    train, valid, syms, tags = prepare(trainpath, testpath)
    train_input, train_len, train_label = instances_to_tensors(train)
    shape = train_input.shape
    for i in range(len(train)):
      label = train[i][0]
      seq = train[i][1]
      assert len(seq) == train_len[i]
      assert label == train_label[i]
      for j in range(shape[1]):
        val = train_input[i][j]
        if j < len(seq):
          assert val > 0
        else:
          assert val == 0

  def testProducer(self):
    trainpath, testpath = self.write_data()
    train, _, _, _ = prepare(trainpath, testpath)
    train_inputs = instances_to_tensors(train)
    xd = train_inputs[0].shape[0]
    yd = train_inputs[0].shape[1]
    config = InMemoryBatchQueueConfig(batch_size=2, num_example=xd, max_len=yd)
    q = InMemoryBatchQueue(config)
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
      sess.run(init_op)
      q.init(train_inputs, sess)
      tf.train.start_queue_runners(sess=sess)
      batch = sess.run(q.batch)
      print(batch)
      batch = sess.run(q.batch)
      print(batch)
      batch = sess.run(q.batch)
      print(batch)

if __name__ == "__main__":
  tf.test.main()
