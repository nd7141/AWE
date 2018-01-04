'''
Tensorflow implementation of PV-DM algorithm as a scikit-learn like model 
with fit, transform methods.

@author: Zichen Wang (wangzc921@gmail.com)
@author: Sergey Ivanov (sergei.ivanov@skolkovotech.ru


@references:

https://github.com/wangz10/tensorflow-playground/blob/master/doc2vec.py
http://arxiv.org/abs/1405.4053
'''
from __future__ import absolute_import
from __future__ import print_function

import os
import math
import random
import json
import collections
from itertools import compress

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
# from sklearn.metrics.pairwise import pairwise_distances

# from word2vec import build_dataset
from main import Graph2Vec

# Set random seeds
SEED = 2018
random.seed(SEED)
np.random.seed(SEED)


class Doc2Vec(BaseEstimator, TransformerMixin):

    def __init__(self, batch_size=128,
                 window_size=8,
                 concat=False,
                 embedding_size_w=64,
                 embedding_size_d=64,
                 loss_type='sampled_softmax_loss',
                 n_neg_samples=64,
                 optimize='Adagrad',
                 learning_rate=1.0,
                 root = '../',
                 ext = 'graphml',
                 steps = 6,
                 epochs = 1,
                 samples_per_node = 1,
                 dataset = 'imdb_b'):

        # bind params to class
        self.batch_size = batch_size
        self.window_size = window_size
        self.concat = concat
        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.loss_type = loss_type
        self.n_neg_samples = n_neg_samples
        self.optimize = optimize
        self.learning_rate = learning_rate

        self.ROOT = root
        self.ext = ext
        self.steps = steps
        self.epochs = epochs
        self.samples_per_node = samples_per_node
        self.dataset = dataset


        # get all graph filenames (document size)
        self.folder = self.ROOT + self.dataset + '/'
        folder_graphs = filter(lambda g: g.endswith(max(self.ext, '')), os.listdir(self.folder))

        self.sorted_graphs = sorted(folder_graphs, key=lambda g: int(g.split('.')[0][5:]))
        self.document_size = len(self.sorted_graphs)
        print(self.document_size)

        # get all AW (vocabulary size)
        self.g2v = Graph2Vec()
        self.g2v._all_paths(self.steps, keep_last=True)
        self.walk_ids = dict()
        for i, path in enumerate(self.g2v.paths[self.steps]):
            self.walk_ids[tuple(path)] = i
        self.vocabulary_size = max(self.walk_ids.values()) + 1
        print(self.walk_ids)
        print(self.vocabulary_size)



        # init all variables in a tensorflow graph
        self._init_graph()

        # create a session
        self.sess = tf.Session(graph=self.graph)

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(SEED)

            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size+1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])
            # Variables.
            # embeddings for words, W in paper
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))

            # embedding for documents (can be sentences or paragraph), D in paper
            self.doc_embeddings = tf.Variable(
                tf.random_uniform([self.document_size, self.embedding_size_d], -1.0, 1.0))

            if self.concat: # concatenating word vectors and doc vector
                combined_embed_vector_length = self.embedding_size_w * self.window_size + self.embedding_size_d
            else: # concatenating the average of word vectors and the doc vector
                combined_embed_vector_length = self.embedding_size_w + self.embedding_size_d

            # softmax weights, W and D vectors should be concatenated before applying softmax
            self.weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, combined_embed_vector_length],
                    stddev=1.0 / math.sqrt(combined_embed_vector_length)))
            # softmax biases
            self.biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            # shape: (batch_size, embeddings_size)
            embed = [] # collect embedding matrices with shape=(batch_size, embedding_size)
            if self.concat:
                for j in range(self.window_size):
                    embed_w = tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                    embed.append(embed_w)
            else:
                # averaging word vectors
                embed_w = tf.zeros([self.batch_size, self.embedding_size_w])
                for j in range(self.window_size):
                    embed_w += tf.nn.embedding_lookup(self.word_embeddings, self.train_dataset[:, j])
                embed.append(embed_w)

            embed_d = tf.nn.embedding_lookup(self.doc_embeddings, self.train_dataset[:, self.window_size])
            embed.append(embed_d)
            # concat word and doc vectors
            self.embed = tf.concat(embed, 1)

            # Compute the loss, using a sample of the negative labels each time.
            if self.loss_type == 'sampled_softmax_loss':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.train_labels,
                                                  self.embed, self.n_neg_samples, self.vocabulary_size)
            elif self.loss_type == 'nce_loss':
                loss= tf.nn.nce_loss(self.weights, self.biases, self.train_labels,
                                     self.embed, self.n_neg_samples, self.vocabulary_size)
            self.loss = tf.reduce_mean(loss)

            # Optimizer.
            if self.optimize == 'Adagrad':
                self.optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss)
            elif self.optimize == 'SGD':
                self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings / norm_w

            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings / norm_d

            # init op
            self.init_op = tf.global_variables_initializer()
            # create a saver
            self.saver = tf.train.Saver()

    def fit(self):

        # with self.sess as session:
        session = self.sess

        session.run(self.init_op)

        average_loss = 0
        global_step = 0
        print('Initialized')
        for _ in range(self.epochs):
            print('Epoch: {}'.format(_))
            for doc_id, graph_fn in enumerate(self.sorted_graphs):
                self.g2v.read_graphml(self.folder + graph_fn)
                self.g2v.create_random_walk_graph()

                N = len(self.g2v.rw_graph)*self.samples_per_node
                print('Graph {}: {} nodes'.format(doc_id, len(self.g2v.rw_graph)))
                for b in range(N):
                    batch_data, batch_labels = self.g2v.generate_batch_pvdm(batch_size=self.batch_size,
                                                                       window_size=self.window_size,
                                                                       steps=self.steps, walk_ids=self.walk_ids,
                                                                       doc_id=doc_id)
                    feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
                    op, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                    global_step += 1

                    average_loss += l
                    if global_step % 100 == 0:
                        # The average loss is an estimate of the loss over the last 2000 batches.
                        print('Average loss at step %d: %f' % (global_step, average_loss))
                        average_loss = 0

        self.doc_embeddings = session.run(self.normalized_doc_embeddings)

        return self

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
            open(os.path.join(path, 'model_dict.json'), 'wb'),
            ensure_ascii=False)
        json.dump(self.reverse_dictionary,
            open(os.path.join(path, 'model_rdict.json'), 'wb'),
            ensure_ascii=False)

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
        estimator = Doc2Vec(**params)
        estimator._restore(path)
        # evaluate the Variable embeddings and bind to estimator
        estimator.word_embeddings = estimator.sess.run(estimator.normalized_word_embeddings)
        estimator.doc_embeddings = estimator.sess.run(estimator.normalized_doc_embeddings)
        # bind dictionaries
        estimator.dictionary = json.load(open(os.path.join(path_dir, 'model_dict.json'), 'rb'))
        reverse_dictionary = json.load(open(os.path.join(path_dir, 'model_rdict.json'), 'rb'))
        # convert indices loaded from json back to int since json does not allow int as keys
        estimator.reverse_dictionary = {int(key):val for key, val in reverse_dictionary.items()}

        return estimator


if __name__ == '__main__':

    d2v = Doc2Vec(concat=False)
    d2v.fit()
    # print(d2v.doc_embeddings)
    np.savetxt('embeddings/imdb_b/embeddings.txt', d2v.doc_embeddings, fmt = "%10.5f")

    console = []