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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import struct
import argparse
import os
import sys
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('input_path', None, 'Path to read instances from.')
flags.DEFINE_string('output_path', None, 'Path to write TF records to.')

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(values):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def write_records(data, path):
    sequences = data.sequences
    labels = data.labels
    writer = tf.python_io.TFRecordWriter(path)
    for index in range(len(labels)):
        sequence = sequences[index]
        label = labels[index]
        example = tf.train.Example(features=tf.train.Features(feature={
            'sequence': _int64_list_feature(sequence),
            'label': _int64_feature(label)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_records(path):
    dataset = Data([], [])
    it =  tf.python_io.tf_record_iterator(path)
    for ex_str in it:
        ex = example_pb2.Example.FromString(ex_str)
        dataset.labels.append(ex.features.feature['label'].int64_list.value[0])
        dataset.sequences.append(ex.features.feature['sequence'].int64_list.value)
    return dataset

def main(unused_argv):
    # Get the data.
    sequences = [
        [1, 1, 1],
        [2, 2, 2, 2, 2],
        [3, 3],
        [4]
    ]
    labels = [1, 2, 3, 4]
    dataset = Data(sequences, labels)

    # Convert to Examples and write the result to TFRecords.
    write_records(dataset, 'dataset.tfrecords')
    result = read_records('dataset.tfrecords')

    assert result == dataset

if __name__ == '__main__':
  tf.app.run()
