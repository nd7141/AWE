'''
Tensorflow implementation of distributed Anonymous Walks Embeddings (AWE).

@author: Sergey Ivanov (sergei.ivanov@skolkovotech.ru -- Adaptation for graph2vec via AW


@references:

'''
from __future__ import division, print_function

import os
import math
import random
import argparse
import sys
import time
import re
import shutil
import threading
from collections import Counter

import numpy as np
import tensorflow as tf

from AnonymousWalkKernel import AnonymousWalks, GraphKernel, Evaluation

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

SEED = 2018

class AWE(object):
    '''
    Computes distributed Anonymous Walk Embeddings.
    '''
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
                 steps = 7,
                 epochs = 1,
                 batches_per_epoch = 1,
                 candidate_func = None,
                 graph_labels = None,
                 regenerate_corpus = False,
                 neighborhood_size=1):
        '''
        Initialize AWE model.
        :param dataset: name of the dataset and corresponding name of the folder.
        :param batch_size: number of batches per iteration of AWE model.
        :param window_size: number of context words.
        :param concat: Concatenate context words or not.
        :param embedding_size_w: embedding size of word
        :param embedding_size_d: embedding size of document
        :param loss_type: sampled softmax or nce
        :param num_samples: number of (negative) samples for every target word.
        :param optimize: SGD or Adagrad
        :param learning_rate: learning rate of the model
        :param root: root folder of the dataset
        :param ext: extension of files with graphs (e.g. graphml)
        :param steps: length of anonymous walk
        :param epochs: number of epochs for iterations
        :param batches_per_epoch: number of batches per epoch for each graph
        :param candidate_func: None (loguniform by default) or uniform
        :param graph_labels: None, edges, nodes, edges_nodes
        '''

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
        self.graph_labels = graph_labels

        self.ROOT = root
        self.ext = ext
        self.steps = steps
        self.epochs = epochs
        self.dataset = dataset

        self.batches_per_epoch = batches_per_epoch


        # switch to have batches_per_epoch = N for every graph with N nodes
        self.flag2iterations = False
        if batches_per_epoch is None:
            self.flag2iterations = True

        # get all graph filenames (document size)
        self.folder = self.ROOT + self.dataset + '/'
        folder_graphs = filter(lambda g: g.endswith(max(self.ext, '')), os.listdir(self.folder))

        self.sorted_graphs = sorted(folder_graphs, key=lambda g: int(re.findall(r'\d+', g)[0]))
        self.document_size = len(self.sorted_graphs)
        print('Number of graphs: {}'.format(self.document_size))

        print('Generating corpus... ', end='')
        self.corpus_fn_name = '{}.corpus'
        self.regenerate_corpus = regenerate_corpus
        self.neiborhood_size = neighborhood_size
        start2gen = time.time()
        self.generate_corpus()
        print('Finished {}'.format(time.time() - start2gen))

        self.vocabulary_size = max(self.walk_ids.values()) + 1
        print('Number of words: {}'.format(self.vocabulary_size))

        # init all variables in a tensorflow graph
        self._init_graph()

        # create a session
        self.sess = tf.Session(graph=self.graph)

    def generate_corpus(self):
        # get all AW (vocabulary size)
        self.g2v = AnonymousWalks()
        if self.graph_labels is None:
            self.g2v._all_paths(self.steps, keep_last=True)
        elif self.graph_labels == 'nodes':
            self.g2v._all_paths_nodes(self.steps, keep_last=True)
        elif self.graph_labels == 'edges':
            self.g2v._all_paths_edges(self.steps, keep_last=True)
        elif self.graph_labels == 'edges_nodes':
            self.g2v._all_paths_edges_nodes(self.steps, keep_last=True)

        self.walk_ids = dict()
        for i, path in enumerate(self.g2v.paths[self.steps]):
            self.walk_ids[tuple(path)] = i

        self.nodes_per_graphs = dict()

        label_suffix = ''
        if graph_labels is not None:
            label_suffix = '_' + graph_labels

        if self.regenerate_corpus == True or not os.path.exists(self.ROOT + self.dataset + '_corpus' + label_suffix):
            if not os.path.exists(self.ROOT + self.dataset + '_corpus' + label_suffix):
                os.mkdir(self.ROOT + self.dataset + '_corpus' + label_suffix)

            for en, graph_fn in enumerate(self.sorted_graphs):
                if en > 0 and not en%100:
                    print(f"Graph {en}")
                g2v = AnonymousWalks()
                g2v.read_graphml(self.folder + graph_fn)
                self.nodes_per_graphs[en] = len(g2v.graph)


                g2v.write_corpus(self.neiborhood_size, self.walk_ids, steps, self.graph_labels,
                                 self.ROOT + self.dataset + '_corpus{}/{}'.format(label_suffix, self.corpus_fn_name.format(en)))

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device('/cpu:0'):
            tf.set_random_seed(SEED)

            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size, self.window_size+1])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # embeddings for anonymous walks
            self.word_embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size_w], -1.0, 1.0))

            # embedding for graphs
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
            loss = None
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

            # Normalize embeddings
            norm_w = tf.sqrt(tf.reduce_sum(tf.square(self.word_embeddings), 1, keep_dims=True))
            self.normalized_word_embeddings = self.word_embeddings/norm_w

            norm_d = tf.sqrt(tf.reduce_sum(tf.square(self.doc_embeddings), 1, keep_dims=True))
            self.normalized_doc_embeddings = self.doc_embeddings/norm_d

            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()

    def _train_thread_body(self):
        '''Train model on random anonymous walk batches.'''
        label_suffix = ''
        if self.graph_labels is not None:
            label_suffix = '_' + graph_labels

        while True:
            batch_data, batch_labels = self.g2v.generate_file_batch(batch_size, window_size, self.doc_id,
                                                                    self.ROOT + self.dataset + '_corpus{}/{}'.format(label_suffix, self.corpus_fn_name.format(self.doc_id)),
                                                                    self.nodes_per_graphs[self.doc_id])
            # batch_data, batch_labels = self.g2v.generate_random_batch(batch_size=self.batch_size,
            #                                                         window_size=self.window_size,
            #                                                         steps=self.steps, walk_ids=self.walk_ids,
            #                                                         doc_id=self.doc_id,
            #                                                         graph_labels = self.graph_labels)
            feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
            op, l = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
            self.sample += 1
            self.global_step += 1

            self.average_loss += l
            # The average loss is an estimate of the loss over the last 100 batches.
            # if self.global_step % 100 == 0:
                # print('Average loss at step %d: %f' % (self.global_step, self.average_loss))
                # self.average_loss = 0

            if self.sample >= self.batches_per_epoch:
                break

    def train(self):
        '''Train the model.'''
        session = self.sess

        session.run(self.init_op)

        self.average_loss = 0
        self.global_step = 0
        print('Initialized')
        random_order = list(range(len(self.sorted_graphs)))
        random.shuffle(random_order)
        for ep in range(self.epochs):
            print('Epoch: {}'.format(ep))
            time2epoch = time.time()
            for rank_id, doc_id in enumerate(random_order):
            # for doc_id, graph_fn in enumerate(self.sorted_graphs):
            #     graph_fn = self.sorted_graphs[doc_id]

                time2graph = time.time()
                self.sample = 0
                self.doc_id = doc_id
                # self.g2v.read_graphml(self.folder + graph_fn)
                # self.g2v.create_random_walk_graph()

                # print('{}-{}. Graph-{}: {} nodes'.format(ep, rank_id, doc_id, len(self.g2v.rw_graph)))
                # if self.flag2iterations == True: # take sample of N words per each graph with N nodes
                #     self.batches_per_epoch = len(self.g2v.rw_graph)

                self._train_thread_body()

                if rank_id > 0 and not rank_id%100:
                    print('Graph {}-{}: {:.2f}'.format(ep, rank_id, time.time() - time2graph))
            print('Time for epoch {}: {:.2f}'.format(ep, time.time() - time2epoch))
            # save temporary embeddings
            if not ep%10:
                self.graph_embeddings = session.run(self.normalized_doc_embeddings)
                np.savez_compressed(RESULTS_FOLDER + '/' + dataset +  '/tmp/embeddings_{}.txt'.format(ep), E=self.graph_embeddings)

        self.graph_embeddings = session.run(self.normalized_doc_embeddings)

        return self

if __name__ == '__main__':

    # Set random seeds
    SEED = 2018
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = 'mutag'

    batch_size = 100
    window_size = 16
    embedding_size_w = 128
    embedding_size_d = 128
    num_samples = 10

    concat = False
    loss_type = 'sampled_softmax'
    optimize = 'Adagrad'
    learning_rate = 0.1
    root = '../Datasets/'
    ext = 'graphml'
    steps = 10
    epochs = 100
    batches_per_epoch = 100
    candidate_func = None
    graph_labels = None

    KERNEL = 'rbf'
    RESULTS_FOLDER = 'doc2vec_results/'
    TRIALS = 10  # number of cross-validation

    regenerate_corpus = True
    neighborhood_size = window_size + 1

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
    parser.add_argument('--batches_per_epoch', default=batches_per_epoch, help='Number of iterations per epoch for each graph', type=int)
    parser.add_argument('--candidate_func', default=candidate_func, help='Sampling function for negatives: uniform or loguniform (None, by default)')
    parser.add_argument('--graph_labels', default=graph_labels,
                        help='Graph labels to use (none, nodes, edges, edges_nodes)')
    parser.add_argument('--regenerate_corpus', default=regenerate_corpus, type=bool,
                        help='If regenerate corpus for training. ')
    parser.add_argument('--neighborhood_size', default=neighborhood_size, type=int,
                        help='Number of context words per line.')



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
    batches_per_epoch = args.batches_per_epoch
    RESULTS_FOLDER = args.results_folder
    candidate_func = args.candidate_func
    graph_labels = args.graph_labels

    regenerate_corpus = args.regenerate_corpus
    neighborhood_size = args.neighborhood_size

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    if not os.path.exists(RESULTS_FOLDER + '/' + dataset):
        os.makedirs(RESULTS_FOLDER + '/' + dataset)

    if not os.path.exists(RESULTS_FOLDER + '/' + dataset + '/tmp/'):
        os.makedirs(RESULTS_FOLDER + '/' + dataset + '/tmp/')

    print('DATASET: {}'.format(dataset))
    print('BATCHES PER EPOCH: {}'.format(batches_per_epoch))
    print('BATCH SIZE: {}'.format(batch_size))
    print('WINDOW SIZE: {}'.format(window_size))
    print('NEGATIVES: {}'.format(num_samples))
    print('EPOCHS: {}'.format(epochs))
    print('')
    print('EMBEDDING WORD SIZE: {}'.format(embedding_size_w))
    print('EMBEDDING GRAPH SIZE: {}'.format(embedding_size_w))
    print('LENGTH: {}'.format(steps))
    print('')


    # initialize model
    awe = AWE(dataset = dataset, batch_size = batch_size, window_size = window_size,
                  embedding_size_w = embedding_size_w, embedding_size_d = embedding_size_d,
                  num_samples = num_samples, concat = concat, loss_type = loss_type,
                  optimize = optimize, learning_rate = learning_rate, root = root,
                  ext = ext, steps = steps, epochs = epochs, batches_per_epoch = batches_per_epoch,
                  candidate_func = candidate_func, graph_labels=graph_labels, regenerate_corpus=regenerate_corpus,
              neighborhood_size=neighborhood_size)
    print()
    start2emb = time.time()
    awe.train() # get embeddings
    finish2emb = time.time()
    print()
    print('Time to compute embeddings: {:.2f} sec'.format(finish2emb - start2emb))
    # E = np.load('imdb_b.embeddings.txt')['E']

    gk = GraphKernel()
    gk.embeddings = awe.graph_embeddings
    gk.write_embeddings(RESULTS_FOLDER + '/' + dataset + '/embeddings.txt')
    # gk.embeddings = E

    # read classes for each graph
    y = []
    with open(root + dataset + '/labels.txt') as f:
        for line in f:
            y.extend(list(map(int, line.strip().split())))
    y = np.array(y)

    ### testing on embeddings
    accuracies = []
    for _ in range(10):
        E = gk.embeddings
        idx_train, idx_test = train_test_split(list(range(E.shape[0])), test_size=0.2)
        E_train = E[idx_train, :]
        y_train = y[idx_train]
        E_test = E[idx_test, :]
        y_test = y[idx_test]

        model = svm.SVC(kernel='rbf', C=1, gamma = 1)
        model.fit(E_train, y_train)
        y_predicted = model.predict(E_test)
        accuracies.append(accuracy_score(y_test, y_predicted))
        print('On Embeddings:', accuracy_score(y_test, y_predicted))
    print(np.max(accuracies), np.mean(accuracies), np.std(accuracies))

    ################## Estimate results: Classification Accuracy ########################
    print()
    for KERNEL in ['rbf', 'linear', 'poly']:

        if KERNEL == 'rbf':
            sigma_grid = [0.00001, 0.0001, 0.001, 0.1, 1, 10]
        else:
            sigma_grid = [1]

        # try:
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

            # run k-fold SVM with cross-validation on C
            print('Evaluating Kernel Matrix on SVM...')
            ev = Evaluation(gk.K, y, verbose=False)
            optimal_test_scores = []
            for _ in range(TRIALS):
                print(TRIALS - _, end=' ')
                sys.stdout.flush()
                accs = ev.evaluate()
                optimal_test_scores.extend(accs)
            print()
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

    label_suffix = ''
    if graph_labels is not None:
        label_suffix = '_' + graph_labels
    shutil.rmtree(root + dataset + '_corpus{}'.format(label_suffix))

    console = []