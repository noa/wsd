# Copyright 2015 Google Inc.
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

import math
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def pad(l):
  for b in bins:
    if b >= l: return b
  return bins[-1]

def bin_for(l):
  for i, b in enumerate(bins):
    if b >= l: return i
  return len(bins) - 1

def get_batch(bin_id, batch_size, data_set, offset=None, preset=None):
  """Get a batch of data, training or testing. This assumes data_set is
  already split into bins and bin_id is the selected bin.

  """

  inputs, targets = [], []
  pad_length = bins[bin_id]
  for b in xrange(batch_size):
    if preset is None:
      elem = random.choice(data_set[bin_id])
      if offset is not None and offset + b < len(data_set[bin_id]):
        elem = data_set[bin_id][offset + b]
    else:
      elem = preset
    inpt, targett, inpl, targetl = elem[0], elem[1], [], []
    for inp in inpt:
      inpl.append(inp + [0 for _ in xrange(pad_length - len(inp))])
    if len(inpl) == 1:
      for _ in xrange(height - 1):
        inpl.append([0 for _ in xrange(pad_length)])
    for target in targett:
      targetl.append(target + [0 for _ in xrange(pad_length - len(target))])
    inputs.append(inpl)
    targets.append(targetl)
  res_input = np.array(inputs, dtype=np.int32)
  res_target = np.array(targets, dtype=np.int32)
  assert list(res_input.shape) == [batch_size, pad_length]
  assert list(res_target.shape) == [batch_size]
  return res_input, res_target

def decode(output):
  return [np.argmax(o, axis=1) for o in output]

def accuracy(inpt_t, output, target_t, batch_size, nprint,
             beam_out=None, beam_scores=None):

  """Calculate output accuracy given target."""
  assert nprint < batch_size + 1
  inpt = []
  for h in xrange(inpt_t.shape[1]):
    inpt.extend([inpt_t[:, h, l] for l in xrange(inpt_t.shape[2])])
  target = [target_t[:, 0, l] for l in xrange(target_t.shape[2])]
  def tok(i):
    if rev_vocab and i < len(rev_vocab):
      return rev_vocab[i]
    return str(i - 1)
  def task_print(inp, output, target):
    stop_bound = 0
    print_len = 0
    while print_len < len(target) and target[print_len] > stop_bound:
      print_len += 1
    print_out("    i: " + " ".join([tok(i) for i in inp if i > 0]))
    print_out("    o: " +
              " ".join([tok(output[l]) for l in xrange(print_len)]))
    print_out("    t: " +
              " ".join([tok(target[l]) for l in xrange(print_len)]))
  decoded_target = target
  decoded_output = decode(output)
  # Use beam output if given and score is high enough.
  if beam_out is not None:
    for b in xrange(batch_size):
      if beam_scores[b] >= 10.0:
        for l in xrange(min(len(decoded_output), beam_out.shape[2])):
          decoded_output[l][b] = int(beam_out[b, 0, l])
  total = 0
  errors = 0
  seq = [0 for b in xrange(batch_size)]
  for l in xrange(len(decoded_output)):
    for b in xrange(batch_size):
      if decoded_target[l][b] > 0:
        total += 1
        if decoded_output[l][b] != decoded_target[l][b]:
          seq[b] = 1
          errors += 1
  e = 0  # Previous error index
  for _ in xrange(min(nprint, sum(seq))):
    while seq[e] == 0:
      e += 1
    task_print([inpt[l][e] for l in xrange(len(inpt))],
               [decoded_output[l][e] for l in xrange(len(decoded_target))],
               [decoded_target[l][e] for l in xrange(len(decoded_target))])
    e += 1
  for b in xrange(nprint - errors):
    task_print([inpt[l][b] for l in xrange(len(inpt))],
               [decoded_output[l][b] for l in xrange(len(decoded_target))],
               [decoded_target[l][b] for l in xrange(len(decoded_target))])
  return errors, total, sum(seq)


def safe_exp(x):
  perp = 10000
  x = float(x)
  if x < 100: perp = math.exp(x)
  if perp > 10000: return 10000
  return perp

def calculate_buckets_scale(data_set, buckets):
  """Calculate buckets scales for the given data set."""
  train_bucket_sizes = [len(data_set[b]) for b in range(len(buckets))]
  train_total_size = max(1, float(sum(train_bucket_sizes)))
  train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                         for i in range(len(train_bucket_sizes))]
  return (train_total_size, train_buckets_scale)

def get_bucket_id(train_buckets_scale_c, max_cur_length, bins, data_set):
  """Get a random bucket id."""

  # Choose a bucket according to data distribution. Pick a random number
  # in [0, 1] and use the corresponding interval in train_buckets_scale.

  random_number_01 = np.random.random_sample()
  bucket_id = min([i for i in xrange(len(train_buckets_scale_c))
                   if train_buckets_scale_c[i] > random_number_01])
  while bucket_id > 0 and not data_set[bucket_id]:
    bucket_id -= 1
  for _ in xrange(10 if np.random.random_sample() < 0.9 else 1):
    if bins[bucket_id] > max_cur_length:
      random_number_01 = min(random_number_01, np.random.random_sample())
      bucket_id = min([i for i in xrange(len(train_buckets_scale_c))
                       if train_buckets_scale_c[i] > random_number_01])
      while bucket_id > 0 and not data_set[bucket_id]:
        bucket_id -= 1
  return bucket_id

def initialize_bins(max_length):
  bins = [2 + bin_idx_i for bin_idx_i in range(256)]
  max_length = min(max_length, bins[-1])
  while len(bins) > 1 and bins[-2] >= max_length:
    bins = bins[:-1]
  while len(bins) > 1 and bins[-2] >= max_length:
    bins = bins[:-1]
  return bins

class UtilTest(tf.test.TestCase):
  def test(self):
    return True

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
