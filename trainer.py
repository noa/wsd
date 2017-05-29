# Copyright 2015 Google Inc. All Rights Reserved.
# Copyright 2017 Johns Hopkins University. All Rights Reserved.
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
import threading
import time

import numpy as np
import tensorflow as tf

import program_utils
import data_utils as data

tf.app.flags.DEFINE_float("lr", 0.1, "Learning rate.")
tf.app.flags.DEFINE_float("init_weight", 0.8, "Initial weights deviation.")
tf.app.flags.DEFINE_float("max_grad_norm", 4.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("cutoff", 1.2, "Cutoff at the gates.")
tf.app.flags.DEFINE_float("dropout", 0.1, "Dropout that much.")
tf.app.flags.DEFINE_float("max_sampling_rate", 0.1, "Maximal sampling rate.")
tf.app.flags.DEFINE_float("length_norm", 0.0, "Length normalization.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "Steps per epoch.")
tf.app.flags.DEFINE_integer("nmaps", 64, "Number of floats in each cell.")
tf.app.flags.DEFINE_integer("vec_size", 64, "Size of word vectors.")
tf.app.flags.DEFINE_integer("train_data_size", 1000, "Training examples/len.")
tf.app.flags.DEFINE_integer("max_length", 40, "Maximum length.")
tf.app.flags.DEFINE_integer("random_seed", 125459, "Random seed.")
tf.app.flags.DEFINE_integer("max_target_vocab", 0,
                            "Maximal size of target vocabulary.")
tf.app.flags.DEFINE_integer("decode_offset", 0, "Offset for decoding.")
tf.app.flags.DEFINE_integer("nprint", 0, "How many test examples to print out.")
tf.app.flags.DEFINE_integer("eval_bin_print", 3, "How many bins step in eval.")
tf.app.flags.DEFINE_integer("mode", 0, "Mode: 0-train other-decode.")
tf.app.flags.DEFINE_bool("do_train", True, "If false, only update memory.")
tf.app.flags.DEFINE_bool("simple_tokenizer", False,
                         "If true, tokenize on spaces only, digits are 0.")
tf.app.flags.DEFINE_bool("normalize_digits", True,
                         "Whether to normalize digits with simple tokenizer.")
tf.app.flags.DEFINE_integer("vocab_size", 16, "Joint vocabulary size.")
tf.app.flags.DEFINE_string("data_dir", "/tmp", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "/tmp/", "Directory to store models.")
tf.app.flags.DEFINE_string("test_file_prefix", "", "Files to test (.en,.fr).")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_string("word_vector_file", "",
                           "Optional file with word vectors to start training.")
tf.app.flags.DEFINE_integer("ps_tasks", 0, "Number of ps tasks used.")
tf.app.flags.DEFINE_string("master", "", "Name of the TensorFlow master.")

FLAGS = tf.app.flags.FLAGS

def read_data(source_path, buckets, max_size=None, print_out=True):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    buckets: the buckets to use.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).
      If set to 1, no data will be returned (empty lists of the right form).
    print_out: whether to print out status or not.

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in buckets]
  counter = 0
  if max_size != 1:
    with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      while source and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0 and print_out:
          print("\treading data line {}".format(counter))
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        source_ids, source_len = zero_split(source_ids)

        ## TODO: create targets

        ## END TODO.

        for bucket_id, size in enumerate(buckets):
          if source_len <= size and target_len <= size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline()
  return data_set

global_train_set = []
train_buckets_scale = []

def calculate_buckets_scale(data_set, buckets):
  """Calculate buckets scales for the given data set."""
  train_bucket_sizes = [len(data_set[b]) for b in xrange(len(buckets))]
  train_total_size = max(1, float(sum(train_bucket_sizes)))

  train_buckets_scale.append(
    [sum(train_bucket_sizes[:i + 1]) / train_total_size
     for i in xrange(len(train_bucket_sizes))])

  return train_total_size

def read_data_into_global(source_path, target_path, buckets,
                          max_size=None, print_out=True):
  """Read data into the global variables (can be in a separate thread)."""
  # pylint: disable=global-variable-not-assigned
  global global_train_set, train_buckets_scale
  # pylint: enable=global-variable-not-assigned
  data_set = read_data(source_path, buckets, max_size, print_out)
  global_train_set.append(data_set)
  train_total_size = calculate_buckets_scale(data_set, buckets)
  if print_out:
    print("\tFinished global data reading ({}).".format(train_total_size))

def initialize(sess=None):
  """Initialize data and model."""
  global MAXLEN_F

  # Create training directory if it does not exist.
  if not tf.gfile.IsDirectory(FLAGS.train_dir):
    data.print_out("Creating training directory %s." % FLAGS.train_dir)
    tf.gfile.MkDir(FLAGS.train_dir)

  decode_suffix = ""
  data.log_filename = os.path.join(FLAGS.train_dir, "wsd/log")

  # Set random seed.
  if FLAGS.random_seed > 0:
    seed = FLAGS.random_seed
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

  # Check data sizes.
  # TODO: figure this out
  assert data.bins
  max_length = min(FLAGS.max_length, data.bins[-1])
  while len(data.bins) > 1 and data.bins[-2] >= max_length + EXTRA_EVAL:
    data.bins = data.bins[:-1]
  for p in FLAGS.problem.split("-"):
    if p in ["progeval", "progsynth"]:
      min_length = max(26, min_length)
  assert max_length + 1 > min_length
  while len(data.bins) > 1 and data.bins[-2] >= max_length + EXTRA_EVAL:
    data.bins = data.bins[:-1]

  # Create checkpoint directory if it does not exist.
  checkpoint_dir = os.path.join(FLAGS.train_dir, 'ckpt')
  if not tf.gfile.IsDirectory(checkpoint_dir):
    data.print_out("Creating checkpoint directory %s." % checkpoint_dir)
    tf.gfile.MkDir(checkpoint_dir)

  # Prepare data.
  data.print_out("Preparing data in %s" % FLAGS.data_dir)
  if FLAGS.simple_tokenizer:
    (train, dev, path) = prepare_data(
      FLAGS.data_dir, FLAGS.vocab_size,
      tokenizer=data_utils.space_tokenizer,
      normalize_digits=FLAGS.normalize_digits)
  else:
    (train, dev, path) = prepare_data(
      FLAGS.data_dir, FLAGS.vocab_size)

  # Read data into buckets and compute their sizes.
  vocab, rev_vocab = wmt.initialize_vocabulary(fr_path)
  data.vocab = vocab
  data.rev_vocab = rev_vocab
  data.print_out("Reading development and training data (limit: %d)."
                 % FLAGS.max_train_data_size)
  dev_set = read_data(dev, data.bins)
  def data_read(size, print_out):
    read_data_into_global(train, data.bins, size, print_out)
  data_read(50000, False)
  read_thread_small = threading.Thread(
    name="reading-data-small", target=lambda: data_read(900000, False))
  read_thread_small.start()
  read_thread_full = threading.Thread(
    name="reading-data-full",
    target=lambda: data_read(FLAGS.max_train_data_size, True))
  read_thread_full.start()
  data.print_out("Data reading set up.")

  # Grid-search parameters.
  lr = FLAGS.lr
  init_weight = FLAGS.init_weight
  max_grad_norm = FLAGS.max_grad_norm

  # Create model and initialize it.
  tf.get_variable_scope().set_initializer(
      tf.orthogonal_initializer(gain=1.8 * init_weight))
  max_sampling_rate = FLAGS.max_sampling_rate if FLAGS.mode == 0 else 0.0
  o = FLAGS.vocab_size if FLAGS.max_target_vocab < 1 else FLAGS.max_target_vocab
  def make_ngpu(cur_beam_size, back):
    return ngpu.NeuralGPU(
        FLAGS.nmaps, FLAGS.vec_size, FLAGS.vocab_size, o,
        FLAGS.dropout, max_grad_norm, FLAGS.cutoff, FLAGS.nconvs,
        FLAGS.kw, FLAGS.kh, FLAGS.height, FLAGS.mem_size,
        lr / math.sqrt(FLAGS.num_replicas), min_length + 3, FLAGS.num_gpus,
        FLAGS.num_replicas, FLAGS.grad_noise_scale, max_sampling_rate,
        atrous=FLAGS.atrous, do_rnn=FLAGS.rnn_baseline,
        do_layer_norm=FLAGS.layer_norm, beam_size=cur_beam_size, backward=back)

  if sess is None:
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      model = make_ngpu(beam_size, True)
  else:
    model = make_ngpu(beam_size, True)

  sv = None
  if sess is None:
    # The supervisor configuration has a few overriden options.
    sv = tf.train.Supervisor(logdir=checkpoint_dir,
                             is_chief=True,
                             saver=model.saver,
                             summary_op=None,
                             save_summaries_secs=60,
                             save_model_secs=15 * 60,
                             global_step=model.global_step)

    config = tf.ConfigProto(allow_soft_placement=True)
    sess = sv.PrepareSession(FLAGS.master, config=config)

  data.print_out("Created model. Checkpoint dir %s" % checkpoint_dir)

  # Load model from parameters if a checkpoint exists.
  ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + ".index"):
    data.print_out("Reading model parameters from %s"
                   % ckpt.model_checkpoint_path)
    model.saver.restore(sess, ckpt.model_checkpoint_path)
  elif sv is None:
    sess.run(tf.global_variables_initializer())
    data.print_out("Initialized variables (no supervisor mode).")
  elif FLAGS.task < 1 and FLAGS.mem_size > 0:
    # sess.run(model.mem_norm_op)
    data.print_out("Created new model and normalized mem (on chief).")

  # Return the model and needed variables.
  return (model, beam_model, min_length, max_length, checkpoint_dir,
          (global_train_set, dev_set, en_path, fr_path), sv, sess)

def single_test(bin_id, model, sess, nprint, batch_size, dev, p, print_out=True,
                offset=None, beam_model=None):
  """Test model on test data of length l using the given session."""
  if not dev[p][bin_id]:
    data.print_out("  bin %d (%d)\t%s\tppl NA errors NA seq-errors NA"
                   % (bin_id, data.bins[bin_id], p))
    return 1.0, 1.0, 0.0
  inpt, target = data.get_batch(
      bin_id, batch_size, dev[p], FLAGS.height, offset)
  if FLAGS.beam_size > 1 and beam_model:
    loss, res, new_tgt, scores = m_step(
        model, beam_model, sess, batch_size, inpt, target, bin_id,
        FLAGS.eval_beam_steps, p)
    score_avgs = [sum(s) / float(len(s)) for s in scores]
    score_maxs = [max(s) for s in scores]
    score_str = ["(%.2f, %.2f)" % (score_avgs[i], score_maxs[i])
                 for i in xrange(FLAGS.eval_beam_steps)]
    data.print_out("  == scores (avg, max): %s" % "; ".join(score_str))
    errors, total, seq_err = data.accuracy(inpt, res, target, batch_size,
                                           nprint, new_tgt, scores[-1])
  else:
    loss, res, _, _ = model.step(sess, inpt, target, False)
    errors, total, seq_err = data.accuracy(inpt, res, target, batch_size,
                                           nprint)
  seq_err = float(seq_err) / batch_size
  if total > 0:
    errors = float(errors) / total
  if print_out:
    data.print_out("  bin %d (%d)\t%s\tppl %.2f errors %.2f seq-errors %.2f"
                   % (bin_id, data.bins[bin_id], p, data.safe_exp(loss),
                      100 * errors, 100 * seq_err))
  return (errors, seq_err, loss)

def assign_vectors(word_vector_file, embedding_key, vocab_path, sess):
  """Assign the embedding_key variable from the given word vectors file."""
  # For words in the word vector file, set their embedding at start.
  if not tf.gfile.Exists(word_vector_file):
    data.print_out("Word vector file does not exist: %s" % word_vector_file)
    sys.exit(1)
  vocab, _ = wmt.initialize_vocabulary(vocab_path)
  vectors_variable = [v for v in tf.trainable_variables()
                      if embedding_key == v.name]
  if len(vectors_variable) != 1:
    data.print_out("Word vector variable not found or too many.")
    sys.exit(1)
  vectors_variable = vectors_variable[0]
  vectors = vectors_variable.eval()
  data.print_out("Pre-setting word vectors from %s" % word_vector_file)
  with tf.gfile.GFile(word_vector_file, mode="r") as f:
    # Lines have format: dog 0.045123 -0.61323 0.413667 ...
    for line in f:
      line_parts = line.split()
      # The first part is the word.
      word = line_parts[0]
      if word in vocab:
        # Remaining parts are components of the vector.
        word_vector = np.array(map(float, line_parts[1:]))
        if len(word_vector) != FLAGS.vec_size:
          data.print_out("Warn: Word '%s', Expecting vector size %d, "
                         "found %d" % (word, FLAGS.vec_size,
                                       len(word_vector)))
        else:
          vectors[vocab[word]] = word_vector
  # Assign the modified vectors to the vectors_variable in the graph.
  sess.run([vectors_variable.initializer],
           {vectors_variable.initializer.inputs[1]: vectors})

def print_vectors(embedding_key, vocab_path, word_vector_file):
  """Print vectors from the given variable."""
  _, rev_vocab = wmt.initialize_vocabulary(vocab_path)
  vectors_variable = [v for v in tf.trainable_variables()
                      if embedding_key == v.name]
  if len(vectors_variable) != 1:
    data.print_out("Word vector variable not found or too many.")
    sys.exit(1)
  vectors_variable = vectors_variable[0]
  vectors = vectors_variable.eval()
  l, s = vectors.shape[0], vectors.shape[1]
  data.print_out("Printing %d word vectors from %s to %s."
                 % (l, embedding_key, word_vector_file))
  with tf.gfile.GFile(word_vector_file, mode="w") as f:
    # Lines have format: dog 0.045123 -0.61323 0.413667 ...
    for i in xrange(l):
      f.write(rev_vocab[i])
      for j in xrange(s):
        f.write(" %.8f" % vectors[i][j])
      f.write("\n")

def get_bucket_id(train_buckets_scale_c, max_cur_length, data_set):
  """Get a random bucket id."""
  # Choose a bucket according to data distribution. Pick a random number
  # in [0, 1] and use the corresponding interval in train_buckets_scale.
  random_number_01 = np.random.random_sample()
  bucket_id = min([i for i in xrange(len(train_buckets_scale_c))
                   if train_buckets_scale_c[i] > random_number_01])
  while bucket_id > 0 and not data_set[bucket_id]:
    bucket_id -= 1
  for _ in xrange(10 if np.random.random_sample() < 0.9 else 1):
    if data.bins[bucket_id] > max_cur_length:
      random_number_01 = min(random_number_01, np.random.random_sample())
      bucket_id = min([i for i in xrange(len(train_buckets_scale_c))
                       if train_buckets_scale_c[i] > random_number_01])
      while bucket_id > 0 and not data_set[bucket_id]:
        bucket_id -= 1
  return bucket_id

def train():
  """Train the model."""
  batch_size = FLAGS.batch_size
  (model, beam_model, min_length, max_length, checkpoint_dir,
   (train_set, dev_set, vocab_path), sv, sess) = initialize()

  with sess.as_default():
    quant_op = model.quantize_op
    max_cur_length = min(min_length + 3, max_length)
    prev_acc_perp = [1000000 for _ in xrange(5)]
    prev_seq_err = 1.0
    is_chief = FLAGS.task < 1
    do_report = False

    # Main traning loop.
    while not sv.ShouldStop():
      global_step, max_cur_length, learning_rate = sess.run(
          [model.global_step, model.cur_length, model.lr])
      acc_loss, acc_l1, acc_total, acc_errors, acc_seq_err = 0.0, 0.0, 0, 0, 0
      acc_grad_norm, step_count, step_c1, step_time = 0.0, 0, 0, 0.0

      # For words in the word vector file, set their embedding at start.
      bound1 = FLAGS.steps_per_checkpoint - 1
      if FLAGS.word_vector_file_en and global_step < bound1 and is_chief:
        assign_vectors(FLAGS.word_vector_file, "embedding:0",
                       vocab_path, sess)

      for _ in xrange(FLAGS.steps_per_checkpoint):
        step_count += 1
        step_c1 += 1
        global_step = int(model.global_step.eval())
        p = random.choice(FLAGS.problem.split("-"))
        train_set = global_train_set
        bucket_id = get_bucket_id(train_buckets_scale, max_cur_length,
                                  train_set)

        # Run a step and time it.
        start_time = time.time()
        inp, target = data.get_batch(bucket_id, batch_size, train_set,
                                     FLAGS.height)
        noise_param = math.sqrt(math.pow(global_step + 1, -0.55) *
                                prev_seq_err) * FLAGS.grad_noise_scale
        loss, res, gnorm, _ = model.step(sess, inp, target, FLAGS.do_train,
                                         noise_param)
        step_time += time.time() - start_time
        acc_grad_norm += 0.0 if gnorm is None else float(gnorm)

        # Accumulate statistics.
        acc_loss += loss
        acc_l1 += loss
        errors, total, seq_err = data.accuracy(
            inp, res, target, batch_size, 0, new_target, scores)
        if FLAGS.nprint > 1:
          print "seq_err: ", seq_err
        acc_total += total
        acc_errors += errors
        acc_seq_err += seq_err

        # Report summary every 10 steps.
        if step_count + 3 > FLAGS.steps_per_checkpoint:
          do_report = True  # Don't polute plot too early.
        if is_chief and step_count % 10 == 1 and do_report:
          cur_loss = acc_l1 / float(step_c1)
          acc_l1, step_c1 = 0.0, 0
          cur_perp = data.safe_exp(cur_loss)
          summary = tf.Summary()
          summary.value.extend(
              [tf.Summary.Value(tag="log_perplexity", simple_value=cur_loss),
               tf.Summary.Value(tag="perplexity", simple_value=cur_perp)])
          sv.SummaryComputed(sess, summary, global_step)

      # Normalize and print out accumulated statistics.
      acc_loss /= step_count
      step_time /= FLAGS.steps_per_checkpoint
      acc_seq_err = float(acc_seq_err) / (step_count * batch_size)
      prev_seq_err = max(0.0, acc_seq_err - 0.02)  # No noise at error < 2%.
      acc_errors = float(acc_errors) / acc_total if acc_total > 0 else 1.0
      t_size = float(sum([len(x) for x in train_set])) / float(1000000)
      msg = ("step %d step-time %.2f train-size %.3f lr %.6f grad-norm %.4f"
             % (global_step + 1, step_time, t_size, learning_rate,
                acc_grad_norm / FLAGS.steps_per_checkpoint))
      data.print_out("%s len %d ppl %.6f errors %.2f sequence-errors %.2f" %
                     (msg, max_cur_length, data.safe_exp(acc_loss),
                      100*acc_errors, 100*acc_seq_err))

      # Lower learning rate if we're worse than the last 5 checkpoints.
      acc_perp = data.safe_exp(acc_loss)
      if acc_perp > max(prev_acc_perp[-5:]) and is_chief:
        sess.run(model.lr_decay_op)
      prev_acc_perp.append(acc_perp)

      # Save checkpoint.
      if is_chief:
        checkpoint_path = os.path.join(checkpoint_dir, "wsd.ckpt")
        model.saver.save(sess, checkpoint_path,
                         global_step=model.global_step)

        # Run evaluation.
        bin_bound = 4
        for p in FLAGS.problem.split("-"):
          total_loss, total_err, tl_counter = 0.0, 0.0, 0
          for bin_id in xrange(len(data.bins)):
            if bin_id < bin_bound or bin_id % FLAGS.eval_bin_print == 1:
              err, _, loss = single_test(bin_id, model, sess, FLAGS.nprint,
                                         batch_size * 4, dev_set, p,
                                         beam_model=beam_model)
              if loss > 0.0:
                total_loss += loss
                total_err += err
                tl_counter += 1
          test_loss = total_loss / max(1, tl_counter)
          test_err = total_err / max(1, tl_counter)
          test_perp = data.safe_exp(test_loss)
          summary = tf.Summary()
          summary.value.extend(
              [tf.Summary.Value(tag="test/%s/loss" % p, simple_value=test_loss),
               tf.Summary.Value(tag="test/%s/error" % p, simple_value=test_err),
               tf.Summary.Value(tag="test/%s/perplexity" % p,
                                simple_value=test_perp)])
          sv.SummaryComputed(sess, summary, global_step)

def linearize(output, rev_fr_vocab, simple_tokenizer=None, eos_id=wmt.EOS_ID):
  # If there is an EOS symbol in outputs, cut them at that point (WMT).
  if eos_id in output:
    output = output[:output.index(eos_id)]
  # Print out French sentence corresponding to outputs.
  if simple_tokenizer or FLAGS.simple_tokenizer:
    vlen = len(rev_fr_vocab)
    def vget(o):
      if o < vlen:
        return rev_fr_vocab[o]
      return "UNK"
    return " ".join([vget(o) for o in output])
  else:
    return wmt.basic_detokenizer([rev_fr_vocab[o] for o in output])

def main(_):
  if FLAGS.mode == 0:
    train()
  elif FLAGS.mode == 1:
    evaluate()
  else:
    raise ValueError()

if __name__ == "__main__":
  tf.app.run()
