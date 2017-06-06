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

import gzip
import os
import re
import tarfile
from six.moves import urllib
import numpy as np
import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD     = "_PAD"
_HELDOUT = "_HELDOUT"
_EOS     = "_EOS"
_UNK     = "_UNK"
_START_VOCAB = [_PAD, _HELDOUT, _EOS, _UNK]

PAD_ID     = 0
HELDOUT_ID = 1
EOS_ID     = 2
UNK_ID     = 3

# Regular expressions used to tokenize.
_CHAR_MARKER = "_CHAR_"
_CHAR_MARKER_LEN = len(_CHAR_MARKER)
_SPEC_CHARS = "" + chr(226) + chr(153) + chr(128)
_PUNCTUATION = "][.,!?\"':;%$#@&*+}{|><=/^~)(_`,0123456789" + _SPEC_CHARS + "-"
_WORD_SPLIT = re.compile("([" + _PUNCTUATION + "])")
_OLD_WORD_SPLIT = re.compile("([.,!?\"':;)(])")
_DIGIT_RE = re.compile("\d")

# Data locations
_PTB_URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"

def instances_from_ids(ids):
  for i in range(len(ids)):
    copy_source_ids = list(ids) + [EOS_ID]
    target_id = copy_source_ids[i]
    if target_id in set([PAD_ID, HELDOUT_ID, EOS_ID]):
      continue
    copy_source_ids[i] = HELDOUT_ID
    yield (copy_source_ids, target_id)

def example_generator(source_path, max_examples=None):
  ninstances = 0
  nlines = 0
  tf.logging.info('reading: {}'.format(source_path))
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    source = source_file.readline()
    while source:
      nlines += 1
      if nlines % 10000 == 0:
        tf.logging.info("\treading data line {}".format(nlines))
      source_ids = [int(x) for x in source.split()]
      for instance in instances_from_ids(source_ids):
        ninstances += 1
        yield instance
      if max_examples and ninstances > max_examples:
        return
      source = source_file.readline()

def maybe_download(directory, filename, url):
  """Download filename from url unless it's already in directory."""
  if not tf.gfile.Exists(directory):
    tf.logging.info("Creating directory %s" % directory)
    os.mkdir(directory)
  filepath = os.path.join(directory, filename)
  if not tf.gfile.Exists(filepath):
    tf.logging.info("Downloading %s to %s" % (url, filepath))
    filepath, _ = urllib.request.urlretrieve(url, filepath)
    statinfo = os.stat(filepath)
    tf.logging.info("Successfully downloaded: {}".format(filename))
  return filepath

def get_ptb_train_set(directory):
  train_path = os.path.join(directory, "simple-examples/data/ptb.train.txt")
  if not (tf.gfile.Exists(train_path)):
    corpus_file = maybe_download(directory, "ptb.tgz", _PTB_URL)
    tf.logging.info("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(directory)
  return train_path

def get_ptb_dev_set(directory):
  valid_path = os.path.join(directory, "simple-examples/data/ptb.valid.txt")
  if not (tf.gfile.Exists(valid_path)):
    corpus_file = maybe_download(directory, "ptb.tgz", _PTB_URL)
    tf.logging.info("Extracting tar file %s" % corpus_file)
    with tarfile.open(corpus_file, "r") as corpus_tar:
      corpus_tar.extractall(directory)
  return valid_path

def is_char(token):
  if len(token) > _CHAR_MARKER_LEN:
    if token[:_CHAR_MARKER_LEN] == _CHAR_MARKER:
      return True
  return False

def basic_detokenizer(tokens):
  """Reverse the process of the basic tokenizer below."""
  result = []
  previous_nospace = True
  for t in tokens:
    if is_char(t):
      result.append(t[_CHAR_MARKER_LEN:])
      previous_nospace = True
    elif t == _SPACE:
      result.append(" ")
      previous_nospace = True
    elif previous_nospace:
      result.append(t)
      previous_nospace = False
    else:
      result.extend([" ", t])
      previous_nospace = False
  return "".join(result)

old_style = False

def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  if old_style:
    for space_separated_fragment in sentence.strip().split():
      words.extend(re.split(_OLD_WORD_SPLIT, space_separated_fragment))
    return [w for w in words if w]
  for space_separated_fragment in sentence.strip().split():
    tokens = [t for t in re.split(_WORD_SPLIT, space_separated_fragment) if t]
    first_is_char = False
    for i, t in enumerate(tokens):
      if len(t) == 1 and t in _PUNCTUATION:
        tokens[i] = _CHAR_MARKER + t
        if i == 0:
          first_is_char = True
    if words and words[-1] != _SPACE and (first_is_char or is_char(words[-1])):
      tokens = [_SPACE] + tokens
    spaced_tokens = []
    for i, tok in enumerate(tokens):
      spaced_tokens.append(tokens[i])
      if i < len(tokens) - 1:
        if tok != _SPACE and not (is_char(tok) or is_char(tokens[i+1])):
          spaced_tokens.append(_SPACE)
    words.extend(spaced_tokens)
  return words

def space_tokenizer(sentence):
  return sentence.strip().split()

def create_vocabulary(vocab_path, data_path, tokenizer,
                      max_vocabulary_size=None, normalize_digits=True,
                      lowercase=False, force=False):
  """Create vocabulary file from data file.

  Data file is assumed to contain one sentence per line. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not tf.gfile.Exists(vocab_path) or force:
    tf.logging.info("Creating vocabulary {} from data {}".format(vocab_path,
                                                                 data_path))
    vocab = {}
    with tf.gfile.GFile(data_path, mode="r") as f:
      counter = 0
      for line_in in f:
        line = " ".join(line_in.split())
        counter += 1
        if counter % 10000 == 0:
          tf.logging.info("processing line {}".format(counter))

        tokens = tokenizer(line)
        for w in tokens:
          if lowercase:
            w = w.lower()
          word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1

      sorted_vocab = sorted(vocab, key=vocab.get, reverse=True)

      vocab_list = _START_VOCAB + sorted_vocab
      if max_vocabulary_size and len(vocab_list) > max_vocabulary_size:
        tf.logging.info("{} > {}; truncating vocab".format(
          len(vocab_list),
          max_vocabulary_size
        ))
        vocab_list = vocab_list[:max_vocabulary_size]
      assert len(vocab_list) > 0
      with tf.gfile.GFile(vocab_path, mode="w") as vocab_file:
        for w in vocab_list:
          vocab_file.write(str(w) + "\n")

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if tf.gfile.Exists(vocabulary_path):
    rev_vocab = []
    with tf.gfile.GFile(vocabulary_path, mode="r") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def ids_to_words(ids, rev_vocab):
  if type(ids) == list:
    return [rev_vocab[i] for i in ids]
  elif type(ids) == np.ndarray:
    if len(ids.shape) == 1:
      ret = []
      for i in range(ids.shape[0]):
        ret.append(rev_vocab[ids[i]])
      return ret
    elif len(ids.shape) == 2:
      ret = []
      for row in range(ids.shape[0]):
        ret.append(ids_to_words(ids[row], rev_vocab))
      return ret
    else:
      raise ValueError('unsupported numpy shape: {}'.format(ids.shape))
  return rev_vocab[ids]

def sentence_to_token_ids(sentence, vocabulary, tokenizer,
                          normalize_digits=old_style, lowercase=False):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """
  words = tokenizer(sentence)
  result = []
  for w in words:
    if lowercase:
      w = w.lower()
    if normalize_digits:
      w = re.sub(_DIGIT_RE, "0", w)
    if w in vocabulary:
      result.append(vocabulary[w])
    else:
      result.append(UNK_ID)
  return result

def data_to_token_ids(data_path, target_path, vocabulary_path, tokenizer,
                      normalize_digits=False, lowercase=False, force=False):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not tf.gfile.Exists(target_path) or force:
    tf.logging.info("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    tf.logging.info("{} entries in vocabulary".format(len(vocab)))
    with tf.gfile.GFile(data_path, mode="r") as data_file:
      with tf.gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 10000 == 0:
            tf.logging.info("tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(line, vocab, tokenizer,
                                            normalize_digits=normalize_digits,
                                            lowercase=lowercase)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_ptb_data(data_dir, tokenizer,
                     vocabulary_size=100000,
                     normalize_digits=False,
                     lowercase=False,
                     force=False):
  """ Create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    vocabulary_size: size of the joint vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    A tuple of 3 elements:
      (1) path to the token-ids for training data-set,
      (2) path to the token-ids for development data-set,
      (3) path to the vocabulary file,
  """
  if not vocabulary_size:
    raise ValueError('must provide maximum vocabulary size')

  # Get ptb data to the specified directory.
  train_path = get_ptb_train_set(data_dir)
  tf.logging.info('PTB train set: {}'.format(train_path))
  dev_path = get_ptb_dev_set(data_dir)
  tf.logging.info("PTB dev set: {}".format(dev_path))

  # Create vocabularies of the appropriate sizes.
  vocab_path = os.path.join(data_dir, "vocab.{}.txt".format(vocabulary_size))
  create_vocabulary(vocab_path, train_path, tokenizer,
                    max_vocabulary_size=vocabulary_size,
                    normalize_digits=normalize_digits,
                    lowercase=lowercase, force=force)
  tf.logging.info('Vocabulary path: {}'.format(vocab_path))

  # Create token ids for the training data.
  train_ids_path = train_path + (".%d.ids" % vocabulary_size)
  data_to_token_ids(train_path, train_ids_path, vocab_path,
                    tokenizer, normalize_digits=normalize_digits,
                    lowercase=lowercase, force=force)
  tf.logging.info('PTB train ids path: {}'.format(train_ids_path))

  # Create token ids for the development data.
  dev_ids_path = dev_path + (".%d.ids" % vocabulary_size)
  data_to_token_ids(dev_path, dev_ids_path, vocab_path,
                    tokenizer, normalize_digits=normalize_digits,
                    lowercase=lowercase, force=force)
  tf.logging.info('PTB dev ids path: {}'.format(dev_ids_path))

  return (train_ids_path, dev_ids_path, vocab_path)

def num_lines(path):
  with open(path) as f:
    ret = 0
    for line in f:
      ret += 1
    return ret

class DataTest(tf.test.TestCase):
  def test(self):
    tmpdatadir = tf.test.get_temp_dir()
    train_ids_path, dev_ids_path, vocab_path = prepare_ptb_data(
      tmpdatadir,
      space_tokenizer,
      force=True)

    tf.logging.info('train: {}'.format(train_ids_path))
    tf.logging.info('valid: {}'.format(dev_ids_path))
    tf.logging.info('vocab: {}'.format(vocab_path))

    assert num_lines(train_ids_path) > 0
    assert num_lines(dev_ids_path) > 0
    assert num_lines(vocab_path) > 0

    dataset = example_generator(train_ids_path)
    nex = 0
    for ex in dataset:
      nex += 1

    tf.logging.info('{} examples'.format(nex))

if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.test.main()
