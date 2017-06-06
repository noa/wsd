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

from wsd_utils import prepare_ptb_data
from wsd_utils import example_generator
from wsd_utils import space_tokenizer

class BucketedBatchQueue(object):
  def __init__(self, raw_path, batch_size, is_training=True, name=None,
               force_preprocess=False):
    record_path = raw_path + '.tfrecords'
    tf.logging.info('Raw input data path: {}'.format(raw_path))
    tf.logging.info('Writing TF example records to: {}'.format(record_path))

    # Step 1: Create training instances from raw IDs file and
    # serialize in binary TF record format for fast access.
    if not os.path.exists(record_path):
      g = example_generator(raw_path)
      write_records(g, record_path)

    # Step 2: Create an input queue that reads TF records from one (or
    # more) paths.
    queue = examples_queue(
      data_sources=record_path, # this is a file pattern (or a single path)
      data_fields_to_features={
        'sequence': tf.VarLenFeature(tf.int64),
        'label': tf.FixedLenFeature([1], tf.int64)
      },
      training=is_training # If is_training==True, the examples are shuffled.
    )

    # Step 3: Read the examples bucketed based on sequence lengths.
    lens, batch = batch_examples(queue, batch_size)
    self._batch = (batch['sequence'], lens, batch['label'])

  @property
  def batch(self):
    return self._batch

class DataTest(tf.test.TestCase):
  def testBucketedProducer(self):
    tmpdatadir = tf.test.get_temp_dir()

    # Step 1: Download data, create vocab, and convert words to word
    # IDs.
    train_ids_path, dev_ids_path, vocab_path = prepare_ptb_data(
      tmpdatadir,
      space_tokenizer
    )

    # Step 2: Create batch queue
    q = BucketedBatchQueue(train_ids_path, 4, True, 'train_queue')

    # Step 3: Run queue.
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
      sess.run(init_op)
      with tf.contrib.slim.queues.QueueRunners(sess):
        batch = sess.run(q.batch)
        print(batch)
        batch = sess.run(q.batch)
        print(batch)

if __name__ == "__main__":
  tf.test.main()
