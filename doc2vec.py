'''
Tensorflow implementation of PV-DM algorithm as a scikit-learn like model 
with fit, transform methods.

@author: Zichen Wang (wangzc921@gmail.com)
@author: Sergey Ivanov (sergei.ivanov@skolkovotech.ru -- Adaptation for graph2vec via AW


@references:

https://github.com/wangz10/tensorflow-playground/blob/master/doc2vec.py
http://arxiv.org/abs/1405.4053
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
import math
import random
import json
import argparse
import sys
import time
import threading
import multiprocessing

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

from main import Graph2Vec, GraphKernel, Evaluation

SEED = 2018

class Doc2Vec(BaseEstimator, TransformerMixin):

    def __init__(self,
                 dataset='imdb_b',
                 batch_size=128,
                 window_size=8,
                 concat=False,
                 embedding_size_w=64,
                 embedding_size_d=64,
                 loss_type='sampled_softmax',
                 num_samples=64,
                 optimize='Adagrad',
                 learning_rate=1.0,
                 root = '../',
                 ext = 'graphml',
                 steps = 6,
                 epochs = 1,
                 samples = 1,
                 concurrent_steps = 2,
                 candidate_func = None):

        # bind params to class
        self.batch_size = batch_size
        self.window_size = window_size
        self.concat = concat
        self.embedding_size_w = embedding_size_w
        self.embedding_size_d = embedding_size_d
        self.loss_type = loss_type
        self.num_samples = num_samples
        self.optimize = optimize
        self.learning_rate = learning_rate
        self.candidate_func = candidate_func

        self.ROOT = root
        self.ext = ext
        self.steps = steps
        self.epochs = epochs
        self.dataset = dataset

        self.concurrent_steps = concurrent_steps
        self.samples = samples

        # switch to have samples = N for every graph with N nodes
        self.flag2samples = False
        if samples is None:
            self.flag2samples = True


        # get all graph filenames (document size)
        self.folder = self.ROOT + self.dataset + '/'
        folder_graphs = filter(lambda g: g.endswith(max(self.ext, '')), os.listdir(self.folder))

        self.sorted_graphs = sorted(folder_graphs, key=lambda g: int(g.split('.')[0][5:]))
        self.document_size = len(self.sorted_graphs)
        print('Number of graphs: {}'.format(self.document_size))

        # get all AW (vocabulary size)
        self.g2v = Graph2Vec()
        self.g2v._all_paths(self.steps, keep_last=True)
        self.walk_ids = dict()
        for i, path in enumerate(self.g2v.paths[self.steps]):
            self.walk_ids[tuple(path)] = i
        self.vocabulary_size = max(self.walk_ids.values()) + 1
        print('Number of words: {}'.format(self.vocabulary_size))



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

            # choosing negative sampling function
            sampled_values = None # log uniform by default
            if self.candidate_func == 'uniform': # change to uniform
                sampled_values = tf.nn.uniform_candidate_sampler(
                    true_classes=tf.to_int64(self.train_labels),
                    num_true=1,
                    num_sampled=self.num_samples,
                    unique=True,
                    range_max=self.vocabulary_size)

            # Compute the loss, using a sample of the negative labels each time.
            if self.loss_type == 'sampled_softmax':
                loss = tf.nn.sampled_softmax_loss(self.weights, self.biases, self.train_labels,
                                                  self.embed,
                                                  self.num_samples,
                                                  self.vocabulary_size,
                                                  sampled_values = sampled_values)
            elif self.loss_type == 'nce':
                loss = tf.nn.nce_loss(self.weights, self.biases, self.train_labels,
                                     self.embed, self.num_samples, self.vocabulary_size,
                                     sampled_values=sampled_values)

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

    def _train_thread_body(self):
        while True:
            batch_data, batch_labels = self.g2v.generate_random_batch(batch_size=self.batch_size,
                                                                    window_size=self.window_size,
                                                                    steps=self.steps, walk_ids=self.walk_ids,
                                                                    doc_id=self.doc_id)
            feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
            op, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            self.sample += 1
            self.global_step += 1

            # print('Thread: {}, Doc-id: {}, Samples: {}/{}'.format(threading.currentThread().getName(), self.doc_id, self.sample, self.samples))

            self.average_loss += l
            if self.global_step % 100 == 0:
                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step %d: %f' % (self.global_step, self.average_loss))
                self.average_loss = 0

            if self.sample >= self.samples:
                break

    def train(self):
        # with self.sess as session:
        session = self.sess

        session.run(self.init_op)

        self.average_loss = 0
        self.global_step = 0
        print('Initialized')
        for _ in range(self.epochs):
            print('Epoch: {}'.format(_))
            for doc_id, graph_fn in enumerate(self.sorted_graphs):
                time2graph = time.time()
                self.sample = 0
                self.doc_id = doc_id
                self.g2v.read_graphml(self.folder + graph_fn)
                self.g2v.create_random_walk_graph()

                print('Graph {}: {} nodes'.format(doc_id, len(self.g2v.rw_graph)))
                if self.flag2samples == True: # take sample of N words per each graph with N nodes
                    self.samples = len(self.g2v.rw_graph)

                self._train_thread_body()

                if doc_id % 10 == 0:
                    print('Time: {}'.format(time.time() - time2graph))

                # workers = []
                # for _ in range(self.concurrent_steps):
                #     # t = threading.Thread(target=self._train_thread_body)
                #     p = multiprocessing.Process(target=self._train_thread_body)
                #     workers.append(p)
                #     p.start()
                #
                # for p in workers:
                #     p.join()

        self.doc_embeddings = session.run(self.normalized_doc_embeddings)

        return self

if __name__ == '__main__':

    # Set random seeds
    SEED = 2018
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = 'imdb_b'

    batch_size = 100
    window_size = 16
    embedding_size_w = 128
    embedding_size_d = 128
    num_samples = 64

    concat = False
    loss_type = 'sampled_softmax'
    optimize = 'Adagrad'
    learning_rate = 1.0
    root = '../'
    ext = 'graphml'
    steps = 7
    epochs = 1
    samples = 100
    concurrent_steps = 2
    candidate_func = None

    KERNEL = 'rbf'
    RESULTS_FOLDER = 'doc2vec_results2/'
    TRIALS = 10  # number of cross-validation

    parser = argparse.ArgumentParser(description='Getting classification accuracy for Graph Kernel Methods')

    parser.add_argument('--dataset', default=dataset, help='Dataset with graphs to classify')

    parser.add_argument('--batch_size', default=batch_size, help='Number of target words in a batch', type=int)
    parser.add_argument('--window_size', default=window_size, help='Number of context words for target', type=int)
    parser.add_argument('--embedding_size_w', default=embedding_size_w, help='Dimension of word embeddings', type=int)
    parser.add_argument('--embedding_size_d', default=embedding_size_d, help='Dimension of document embeddings', type=int)
    parser.add_argument('--num_samples', default=num_samples, help='Number of (negative) samples for objective functions', type=int)

    parser.add_argument('--concat', default=concat, help='Concatenate or Average context words', type=bool)
    parser.add_argument('--loss_type', default=loss_type, help='sampled_softmax or nce')
    parser.add_argument('--optimize', default=optimize, help='Adagrad or SGD')
    parser.add_argument('--learning_rate', default=learning_rate, help='Learning rate of optimizer')
    parser.add_argument('--root', default=root, help='Root folder of dataset')
    parser.add_argument('--ext', default=ext, help='Extension of graph filenames')
    parser.add_argument('--results_folder', default=RESULTS_FOLDER, help='Folder to store results')

    parser.add_argument('--steps', default=steps, help='Number of steps for AW', type=int)
    parser.add_argument('--epochs', default=epochs, help='Number of epochs to train', type=int)
    parser.add_argument('--samples', default=samples, help='Number of samples for each graph', type=int)
    parser.add_argument('--concurrent', default=concurrent_steps, help='Number of threads', type=int)
    parser.add_argument('--candidate_func', default=candidate_func, help='Sampling function for negatives: uniform or loguniform (None, by default)')


    args = parser.parse_args()

    dataset = args.dataset

    batch_size = args.batch_size
    window_size = args.window_size
    embedding_size_w = args.embedding_size_w
    embedding_size_d = args.embedding_size_d
    num_samples = args.num_samples

    concat = args.concat
    loss_type = args.loss_type
    optimize = args.optimize
    learning_rate = args.learning_rate
    root = args.root
    ext = args.ext
    steps = args.steps
    epochs = args.epochs
    samples = args.samples
    concurrent_steps = args.concurrent
    RESULTS_FOLDER = args.results_folder
    candidate_func = args.candidate_func

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    if not os.path.exists(RESULTS_FOLDER + '/' + dataset):
        os.makedirs(RESULTS_FOLDER + '/' + dataset)

    print('DATASET: {}'.format(dataset))
    print('BATCH SIZE: {}'.format(batch_size))
    print('SAMPLES: {}'.format(samples))
    print('WINDOW SIZE: {}'.format(window_size))
    print('')
    print('EMBEDDING WORD SIZE: {}'.format(embedding_size_w))
    print('EMBEDDING DOCUMENT SIZE: {}'.format(embedding_size_w))
    print('STEPS: {}'.format(steps))
    print('')


    # initialize model
    d2v = Doc2Vec(dataset = dataset, batch_size = batch_size, window_size = window_size,
                  embedding_size_w = embedding_size_w, embedding_size_d = embedding_size_d,
                  num_samples = num_samples, concat = concat, loss_type = loss_type,
                  optimize = optimize, learning_rate = learning_rate, root = root,
                  ext = ext, steps = steps, epochs = epochs, samples = samples, concurrent_steps=concurrent_steps,
                  candidate_func = candidate_func)
    print()
    start2emb = time.time()
    d2v.train() # get embeddings
    finish2emb = time.time()
    print()
    print('Time to compute embeddings: {:.2f} sec'.format(finish2emb - start2emb))
    # E = np.load(RESULTS_FOLDER + '/' + dataset + '/embeddings.txt.npz')['E']

    gk = GraphKernel()
    gk.embeddings = d2v.doc_embeddings
    # gk.embeddings = E
    gk.write_embeddings(RESULTS_FOLDER + '/' + dataset + '/embeddings.txt')

    # read labels for each graph
    y = []
    with open('../reddit_m5k/labels.txt') as f:
        for line in f:
            y.extend(list(map(int, line.strip().split())))
    y = np.array(y)


    ################## Estimate results: Classification Accuracy ########################
    print()
    for KERNEL in ['rbf', 'dot', 'poly']:

        if KERNEL == 'rbf':
            sigma_grid = [1]
        else:
            sigma_grid = [1]

        # try:
        # cross-validation on sigma
        for s_ix in range(len(sigma_grid)):
            print('Setup: ',dataset, KERNEL, sigma_grid[s_ix])
            sys.stdout.flush()

            print('Computing Kernel Matrix...')
            start2kernelmatrix = time.time()
            gk.kernel_matrix(kernel_method=KERNEL, build_embeddings=False, sigma=sigma_grid[s_ix])
            finish2kernelmatrix = time.time()
            print('Time to compute Kernel Matrix: ', finish2kernelmatrix - start2kernelmatrix)
            sys.stdout.flush()

            # write kernel matrix and embeddings
            # gk.write_kernel_matrix('{}/{}/kernel_{}_{}.txt'.format(RESULTS_FOLDER, dataset, KERNEL, sigma_grid[s_ix]))
            # dump = np.load('{}/{}/kernel_{}_{}.txt.npz'.format(RESULTS_FOLDER, dataset, KERNEL, sigma_grid[s_ix]))
            # gk.K = dump['K']

            N, M = gk.K.shape
            print('Kernel matrix shape: {}x{}'.format(N, M))
            sys.stdout.flush()

            print('Evaluating Kernel Matrix on SVM...')
            # run k-fold SVM with cross-validation on C
            ev = Evaluation(gk.K, y, verbose=False)
            optimal_test_scores = []
            for _ in range(TRIALS):
                print(TRIALS - _, end=' ')
                sys.stdout.flush()
                accs = ev.evaluate()
                optimal_test_scores.extend(accs)
            print()


            # optimal_val_scores = []
            # optimal_test_scores = []
            # for _ in range(TRIALS):
            #     start2SVM = time.time()
            #     val, test, C = gk.run_SVM(y, alpha=.9, features='kernels')
            #     finish2SVM = time.time()
            #     print('{} Time to run SVM: {:.2f}'.format(_, finish2SVM - start2SVM))
            #     optimal_val_scores.append(val)
            #     optimal_test_scores.append(test)
            #     print(val, test, C)
            #
            # print('Average Performance on Validation:', np.mean(optimal_val_scores))
            print('Average Performance on Test: {:.2f}% +-{:.2f}%'.format(100*np.mean(optimal_test_scores),
                                                                          100*np.std(optimal_test_scores)))
            sys.stdout.flush()
            # append results of dataset to the file
            with open('{}/{}/performance_{}_{}_{}.txt'.format(RESULTS_FOLDER, dataset, dataset, KERNEL, steps), 'a') as f:
                f.write('{} {} {} {} {} {} {}\n'.format(dataset, KERNEL, steps, sigma_grid[s_ix],
                                                           np.mean(optimal_test_scores), np.std(optimal_test_scores),
                                                              finish2kernelmatrix - start2kernelmatrix))
            print()
        # except Exception as e:
        #     print('ERROR FOR', dataset, KERNEL, steps)
        #     raise e

    console = []