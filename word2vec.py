'''
Tensorflow implementation of word2vec algorithm as a scikit-learn like model 
with fit, transform methods.

@author: Zichen Wang (wangzc921@gmail.com)

@references:

https://www.kaggle.com/c/word2vec-nlp-tutorial/details/
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
https://github.com/wangz10/UdacityDeepLearning/blob/master/5_word2vec.ipynb
'''

from __future__ import absolute_import
from __future__ import print_function

import collections
import math
import os
import random
import re
import json
from itertools import compress

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics.pairwise import pairwise_distances

# Set random seeds
SEED = 2016
random.seed(SEED)
np.random.seed(SEED)


#################### Util functions #################### 

def build_dataset(words, vocabulary_size=50000):
    '''
    Build the dictionary and replace rare words with UNK token.

    Parameters
    ----------
    words: list of tokens
    vocabulary_size: maximum number of top occurring tokens to produce,
        rare tokens will be replaced by 'UNK'
    '''
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict() # {word: index}
    for word, _ in count:
        dictionary[word] = len(dictionary)
        data = list() # collect index
        unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count # list of tuples (word, count)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary



data_index = 0

def generate_batch_skipgram(data, batch_size, num_skips, skip_window):
    '''
    Batch generator for skip-gram model.

    Parameters
    ----------
    data: list of index of words
    batch_size: number of words in each mini-batch
    num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
    skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span) # used for collecting data[data_index] in the sliding window
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


def generate_batch_cbow(data, batch_size, num_skips, skip_window):
    '''
    Batch generator for CBOW (Continuous Bag of Words).
    batch should be a shape of (batch_size, num_skips)

    Parameters
    ----------
    data: list of index of words
    batch_size: number of words in each mini-batch
    num_skips: number of surrounding words on both direction (2: one word ahead and one word following)
    skip_window: number of words at both ends of a sentence to skip (1: skip the first and last word of a sentence)
    '''
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size, num_skips), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span) # used for collecting data[data_index] in the sliding window
    # collect the first window of words
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    # move the sliding window
    for i in range(batch_size):
        mask = [1] * span
        mask[skip_window] = 0
        batch[i, :] = list(compress(buffer, mask)) # all surrounding words
        labels[i, 0] = buffer[skip_window] # the word at the center
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


class Word2Vec(BaseEstimator, TransformerMixin):

    def __init__(self, batch_size=128, num_skips=2, skip_window=1,
        architecture='skip-gram', embedding_size=128, vocabulary_size=50000,
        loss_type='sampled_softmax_loss', n_neg_samples=64,
        optimize='Adagrad',
        learning_rate=1.0, n_steps=100001,
        valid_size=16, valid_window=100):
        # bind params to class
        self.batch_size = batch_size
        self.num_skips = num_skips
        self.skip_window = skip_window
        self.architecture = architecture
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.loss_type = loss_type
        self.n_neg_samples = n_neg_samples
        self.optimize = optimize
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.valid_size = valid_size
        self.valid_window = valid_window
        # pick a list of words as validataion set
        self._pick_valid_samples()
        # choose a batch_generator function for feed_dict
        self._choose_batch_generator()
        # init all variables in a tensorflow graph
        self._init_graph()

        # create a session
        self.sess = tf.Session(graph=self.graph)
        # self.sess = tf.InteractiveSession(graph=self.graph)


    def _pick_valid_samples(self):
        valid_examples = np.array(random.sample(range(self.valid_window), self.valid_size))
        self.valid_examples = valid_examples

    def _choose_batch_generator(self):
        if self.architecture == 'skip-gram':
            self.generate_batch = generate_batch_skipgram
        elif self.architecture == 'cbow':
            self.generate_batch = generate_batch_cbow

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(SEED)
            # Input data.
            if self.architecture == 'skip-gram':
                self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
            elif self.architecture == 'cbow':
                self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_skips])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            self.valid_dataset = tf.constant(self.valid_examples, dtype=tf.int32)

            # Variables.
            self.embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))

            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                    stddev=1.0 / math.sqrt(self.embedding_size)))

            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))
  
            # Model.
            # Look up embeddings for inputs.
            if self.architecture == 'skip-gram':
                self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_dataset)
            elif self.architecture == 'cbow':
                embed = tf.zeros([self.batch_size, self.embedding_size])
                for j in range(self.num_skips):
                    embed += tf.nn.embedding_lookup(self.embeddings, self.train_dataset[:, j])
                self.embed = embed

            # Compute the loss, using a sample of the negative labels each time.
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.embed,
                    self.train_labels, self.n_neg_samples, self.vocabulary_size)
            elif self.loss_type == 'nce_loss':
                loss= tf.nn.nce_loss(self.weights, self.biases, self.embed,
                    self.train_labels, self.n_neg_samples, self.vocabulary_size)
            self.loss = tf.reduce_mean(loss)

            # Optimizer.
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
  
            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(self.embeddings), 1, keep_dims=True))
            self.normalized_embeddings = self.embeddings / norm
            self.valid_embeddings = tf.nn.embedding_lookup(
                self.normalized_embeddings, self.valid_dataset)
            self.similarity = tf.matmul(self.valid_embeddings, tf.transpose(self.normalized_embeddings))

            # init op
            self.init_op = tf.initialize_all_variables()
            # create a saver
            self.saver = tf.train.Saver()


    def _build_dictionaries(self, words):
        '''
        Process tokens and build dictionaries mapping between tokens and
        their indices. Also generate token count and bind these to self.
        '''

        data, count, dictionary, reverse_dictionary = build_dataset(words,
            self.vocabulary_size)
        self.dictionary = dictionary
        self.reverse_dictionary = reverse_dictionary
        self.count = count
        return data


    def fit(self, words):
        '''
        words: a list of words.
        '''
        # pre-process words to generate indices and dictionaries
        data = self._build_dictionaries(words)

        # with self.sess as session:
        session = self.sess

        session.run(self.init_op)
        # tf.initialize_all_variables().run()

        average_loss = 0
        print("Initialized")
        for step in range(self.n_steps):
            batch_data, batch_labels = self.generate_batch(data,
                self.batch_size, self.num_skips, self.skip_window)
            feed_dict = {self.train_dataset : batch_data, self.train_labels : batch_labels}
            op, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
            average_loss += l
            if step % 2000 == 0:
                if step > 0:
                    average_loss = average_loss / 2000
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (step, average_loss))
                average_loss = 0
            # note that this is expensive (~20% slowdown if computed every 500 steps)
            # if step % 10000 == 0:
            # 	sim = similarity.eval()
            # 	for i in range(valid_size):
            # 		valid_word = reverse_dictionary[valid_examples[i]]
            # 		top_k = 8 # number of nearest neighbors
            # 		nearest = (-sim[i, :]).argsort()[1:top_k+1]
            # 		log = 'Nearest to %s:' % valid_word
            # 		for k in range(top_k):
            # 			close_word = reverse_dictionary[nearest[k]]
            # 			log = '%s %s,' % (log, close_word)
            # 		print(log)

        # final_embeddings = self.normalized_embeddings.eval()
        final_embeddings = session.run(self.normalized_embeddings)
        self.final_embeddings = final_embeddings

        return self

    def transform(self, words):
        '''
        Look up embedding vectors using indices
        words: list of words
        '''
        # make sure all word index are in range
        try:
            indices = [self.dictionary[w] for w in words]
        except KeyError:
            raise KeyError('Some word(s) not in dictionary')
        else:
            return self.final_embeddings[indices]


    def sort(self, word):
        '''
        Use an input word to sort words using cosine distance in ascending order
        '''
        assert word in self.dictionary
        i = self.dictionary[word]
        vec = self.final_embeddings[i].reshape(1, -1)
        # Calculate pairwise cosine distance and flatten to 1-d
        pdist = pairwise_distances(self.final_embeddings, vec, metric='cosine').ravel()
        return [self.reverse_dictionary[i] for i in pdist.argsort()]


    def save(self, path):
        '''
        To save trained model and its params.
        '''
        save_path = self.saver.save(self.sess,
            os.path.join(path, 'model.ckpt'))
        # save parameters of the model
        params = self.get_params()
        json.dump(params,
            open(os.path.join(path, 'model_params.json'), 'wb'))

        # save dictionary, reverse_dictionary
        json.dump(self.dictionary,
            open(os.path.join(path, 'model_dict.json'), 'wb'))
        json.dump(self.reverse_dictionary,
            open(os.path.join(path, 'model_rdict.json'), 'wb'))

        print("Model saved in file: %s" % save_path)
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

    @classmethod
    def restore(cls, path):
        '''
        To restore a saved model.
        '''
        # load params of the model
        path_dir = os.path.dirname(path)
        params = json.load(open(os.path.join(path_dir, 'model_params.json'), 'rb'))
        # init an instance of this class
        estimator = Word2Vec(**params)
        estimator._restore(path)
        # evaluate the Variable normalized_embeddings and bind to final_embeddings
        estimator.final_embeddings = estimator.sess.run(estimator.normalized_embeddings)
        # bind dictionaries
        estimator.dictionary = json.load(open(os.path.join(path_dir, 'model_dict.json'), 'rb'))
        reverse_dictionary = json.load(open(os.path.join(path_dir, 'model_rdict.json'), 'rb'))
        # convert indices loaded from json back to int since json does not allow int as keys
        estimator.reverse_dictionary = {int(key):val for key, val in reverse_dictionary.items()}

        return estimator



