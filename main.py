#!/opt/conda/bin/python

import networkx as nx
import random, time, math, os, sys
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import argparse

class Graph2Vec(object):
    '''
    Computes Anonymous Walk Embeddings of a Graph.
    '''
    def __init__(self, G = None):
        self._graph = G
        # paths are dictionary between step and all-paths
        self.paths = dict()
        self.__methods = ['sampling', 'exact']

    @property
    def graph(self):
        return self._graph

    @graph.setter
    def graph(self, G):
        self._graph = G

    def read_graph_from_text(self, filename, header = True, weights = True, sep = ',', directed = False):
        '''Read from Text Files.'''
        G = nx.Graph()
        if directed:
            G = nx.DiGraph()
        with open(filename) as f:
            if header:
                next(f)
            for line in f:
                splitted = line.strip().split(sep)
                u = splitted[0]
                v = splitted[1]
                G.add_edge(u, v)
                if weights:
                    w = float(splitted[2])
                    G[u][v]['weight'] = w
        self.graph = G
        return self.graph

    def read_graphml(self, filename):
        self.graph = nx.read_graphml(filename)
        return self.graph

    def create_random_walk_graph(self):
        '''Creates a probabilistic graph from graph with weights.'''
        if self.graph is None:
            raise ValueError("You should first create a weighted graph.")

        # get name of the label on graph edges (assume all label names are the same)
        label_name = 'weight'
        # for e in self.graph.edges_iter(data=True):
        #     label_name = e[2].keys()[0]
        #     break

        RW = nx.DiGraph()
        for node in self.graph:
            edges = self.graph[node]
            total = float(sum([edges[v].get(label_name, 1) for v in edges if v != node]))
            for v in edges:
                if v != node:
                    RW.add_edge(node, v, {'weight': edges[v].get(label_name,1) / total})
        self.rw_graph = RW

    def _all_paths(self, steps, keep_last = False):
        '''Get all possible meta-paths of length up to steps.'''
        paths = []
        last_step_paths = [[0, 1]]
        for i in range(2, steps+1):
            current_step_paths = []
            for j in range(i + 1):
                for walks in last_step_paths:
                    if walks[-1] != j and j <= max(walks) + 1:
                        paths.append(walks + [j])
                        current_step_paths.append(walks + [j])
            last_step_paths = current_step_paths
        # filter only on n-steps walks
        if keep_last:
            paths = filter(lambda path: len(path) ==  steps + 1, paths)
        self.paths[steps] = paths

    def _all_paths_edges(self, steps):
        '''Get all possible meta-paths of length up to steps, using edge labels'''
        paths = []
        last_step_paths = [[]]
        for i in range(0, steps):
            current_step_paths = []
            for j in range(i + 1):
                for walks in last_step_paths:
                    if j <= max(walks + [0]) + 1:
                        paths.append(walks + [j])
                        current_step_paths.append(walks + [j])
            last_step_paths = current_step_paths
        self.paths[steps] = paths
        return paths

    def _all_paths_nodes(self, steps):
        '''Get all possible meta-paths of length up to steps, using node labels'''
        paths = []
        last_step_paths = [[0]]
        for i in range(1, steps+1):
            current_step_paths = []
            for j in range(i + 1):
                for walks in last_step_paths:
                    if j <= max(walks) + 1:
                        paths.append(walks + [j])
                        current_step_paths.append(walks + [j])
            last_step_paths = current_step_paths
        self.paths[steps] = paths
        return paths

    def _all_paths_edges_nodes(self, steps):
        '''Get all possible meta-paths of length up to steps, using edge-node labels'''
        edge_paths = self._all_paths_edges(steps)
        node_paths = self._all_paths_nodes(steps)
        paths = []
        for p1 in edge_paths:
            for p2 in node_paths:
                if len(p2) == len(p1) + 1:
                    current_path = [p2[0]]
                    for ix in range(len(p1)):
                        current_path.append(p1[ix])
                        current_path.append(p2[ix+1])
                    paths.append(current_path)
        self.paths[steps] = paths
        return paths

    def walk2pattern(self, walk):
        '''Converts a walk with arbitrary nodes to meta-walk, without considering labels.'''
        idx = 0
        pattern = []
        d = dict()
        for node in walk:
            if node not in d:
                d[node] = idx
                idx += 1
            pattern.append(d[node])
        return tuple(pattern)

    def walk2pattern_edges(self, walk):
        '''Converts a walk with arbitrary nodes to meta-walk, but also considering edge labels.'''
        idx = 0
        pattern = []
        d = dict()
        for ix, node in enumerate(walk[:-1]):
            label = int(self.graph[walk[ix]][walk[ix + 1]]['label'])
            if label not in d:
                d[label] = idx
                idx += 1
            pattern.append(d[label])
        return tuple(pattern)

    def walk2pattern_nodes(self, walk):
        '''Converts a walk with arbitrary nodes to meta-walk, but also considering node labels.'''
        idx = 0
        pattern = []
        d = dict()
        for node in walk:
            label = self.graph.node[node]['label']
            if label not in d:
                d[label] = idx
                idx += 1
            pattern.append(d[label])
        return tuple(pattern)

    def walk2pattern_edges_nodes(self, walk):
        '''Converts a walk with arbitrary nodes to meta-walk, but also considering edge-node labels'''
        node_idx = 0
        edge_idx = 0
        pattern = [0]
        node_labels = dict()
        edge_labels = dict()
        for ix, node in enumerate(walk[1:]):
            node_label = self.graph.node[node]['label']
            edge_label = int(self.graph[walk[ix]][walk[ix+1]]['label'])
            if node_label not in node_labels:
                node_labels[node_label] = node_idx
                node_idx += 1
            if edge_label not in edge_labels:
                edge_labels[edge_label] = edge_idx
                edge_idx += 1
            pattern.append(node_labels[node_label])
            pattern.append(edge_labels[edge_label])
        return tuple(pattern)

    def n_samples(self, steps, delta, eps):
        a = len(self.paths[steps])
        estimation = 2*(math.log(2)*a + math.log(1./delta))/eps**2
        return int(estimation) + 1

    def _random_step_node(self, node):
        '''Moves one step from the current according to probabilities of outgoing edges.
        Return next node.'''
        if self.rw_graph is None:
            raise ValueError("Create a Random Walk graph first with {}".format(self.create_random_walk_graph.__name__))
        r = random.uniform(0, 1)
        low = 0
        for v in self.rw_graph[node]:
            p = self.rw_graph[node][v]['weight']
            if r <= low + p:
                return v
            low += p

    def _random_walk_node(self, node, steps):
        '''Creates a random walk from a node for arbitrary steps.
        Returns a tuple with consequent nodes.'''
        d = dict()
        d[node] = 0
        count = 1
        walk = [d[node]]
        for i in range(steps):
            v = self._random_step_node(node)
            if v not in d:
                d[v] = count
                count += 1
            walk.append(d[v])
            node = v
        return tuple(walk)

    def generate_random_batch(self, batch_size, window_size, steps, walk_ids, doc_id):
        '''
        Generates a (random) batch and labels for doc2vec PV-DM.
        reference: https://arxiv.org/abs/1405.4053

        Batch is a numpy matrix of shape (batch_size, window_size + 1).
        Each row is a context words (AW) that co-occur with a target word.
        To form context, batch generates a sample of random walks and converts then into AW.
        All, except the last AW, are considered to be context words.
        Last AW is considered to be the label.
        The last column of batch corresponds to doc_id, i.e. id of the whole graph.

        :param batch_size: number of samples in the batch.
        :param window_size: number of context words.
        :param steps: number of steps in a random walk. The bigger steps, the more possible words we have. Only walks
        of length = steps are considered for context words.
        :param walk_ids: dictionary between AW and its id. AW corresponds to a possible word.
        :param doc_id: the id of the graph.
        :return: batch (batch_size, window_size + 1) numpy array with batches for doc2vec
                 labels (batch_size, 1) numpy array with target words for doc2vec
        '''

        if self.rw_graph is None:
            raise ValueError("Create a Random Walk graph first with {}".format(self.create_random_walk_graph.__name__))
        if steps not in self.paths:
            raise ValueError("Create all possible AW first with {}".format(self._all_paths.__name__))

        batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
        labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

        batch[:, window_size] = doc_id # last column is for document id

        # create a batch and labels
        i = 0 # number of samples in the batch
        while i < batch_size:
            node = random.choice(self.rw_graph.nodes()) # choose random node
            # generate anonymous walks from this node
            aw = [walk_ids[self._random_walk_node(node, steps)] for _ in range(window_size + 1)]
            batch[i, :window_size] = aw[:window_size]
            labels[i, 0] = aw[window_size]
            i += 1
        return batch, labels

    def generate_graph_batch(self, window_size, steps, walk_ids, doc_id):
        '''
        Generate a batch with N (# nodes) samples (one per node)
        '''

        if self.rw_graph is None:
            raise ValueError("Create a Random Walk graph first with {}".format(self.create_random_walk_graph.__name__))
        if steps not in self.paths:
            raise ValueError("Create all possible AW first with {}".format(self._all_paths.__name__))

        N = len(self.rw_graph) # nodes

        batch = np.ndarray(shape=(N, window_size + 1), dtype=np.int32)
        labels = np.ndarray(shape=(N, 1), dtype=np.int32)

        batch[:, window_size] = doc_id # last column is for document id

        # create a batch and labels
        for i, node in enumerate(self.rw_graph):
            # generate anonymous walks from this node
            aw = [walk_ids[self._random_walk_node(node, steps)] for _ in range(window_size + 1)]
            batch[i, :window_size] = aw[:window_size]
            labels[i, 0] = aw[window_size]
        return batch, labels

    def _sampling(self, steps, MC, prop=True):
        '''Find vector representation using sampling method.
        Run MC random walks for random nodes in the graph.
        steps is the number of steps.
        MC is the number of iterations.
        Returns dictionary pattern to probability.
        '''
        walks = dict()
        N = len(self.rw_graph)
        for it in range(MC):
            node = np.random.choice(self.rw_graph.nodes())
            # run a random walk with n steps, and then look at sub-walks
            w = self._random_walk_node(node, steps)
            for length in range(3, len(w) + 1):
                w_cropped = w[:length]
                amount = 1.
                if prop:
                    amount /= MC
                if w_cropped not in walks:
                    walks[w_cropped] = amount
                else:
                    walks[w_cropped] += amount
        return walks

    def _exact(self, steps, labels = None, prop = True, verbose = True):
        '''Find vector representation using exact method.
            Calculates probabilities from each node to all other nodes within n steps.
            Running time is the O(# number of random walks) <= O(n*d_max^steps).
            labels, possible values None (no labels), 'edges', 'nodes', 'edges_nodes'.
            steps is the number of steps.
            Returns dictionary pattern to probability.
        '''
        walks = dict()
        all_walks = []

        def patterns(RW, node, steps, walks, current_walk=None, current_dist=1.):
            if current_walk is None:
                current_walk = [node]
            if len(current_walk) > 1:  # walks with more than 1 edge
                all_walks.append(current_walk)
                if labels is None:
                    w2p = self.walk2pattern(current_walk)
                elif labels == 'edges':
                    w2p = self.walk2pattern_edges(current_walk)
                elif labels == 'nodes':
                    w2p = self.walk2pattern_nodes(current_walk)
                elif labels == 'edges_nodes':
                    w2p = self.walk2pattern_edges_nodes(current_walk)
                else:
                    raise ValueError('labels argument should be one of the following: edges, nodes, edges_nodes, None.')
                amount = current_dist
                if prop:
                    amount /= len(RW)
                walks[w2p] = walks.get(w2p, 0) + amount # / len(RW) test: not normalizing
            if steps > 0:
                for v in RW[node]:
                    patterns(RW, v, steps - 1, walks, current_walk + [v], current_dist * RW[node][v]['weight'])

        for node in self.rw_graph:
            patterns(self.rw_graph, node, steps, walks)
        if verbose:
            print('Total walks of size {} in a graph:'.format(steps), len(all_walks))
        return walks

    def embed(self, steps, method = 'exact', MC = None, delta = 0.1, eps = 0.1,
              prop=True, labels = None, keep_last = False, verbose = True):
        '''Generic function to get vector representation.
        method can be sampling, exact
        steps is the number of steps.
        MC is the number of iterations.
        labels, possible values None (no labels), 'edges', 'nodes', 'edges_nodes'.
        delta is probability devitation from the true distribution of meta-walks
        eps is absolute value for deviation of first norm
        Return vector and meta information as dictionary.'''

        # Create a random walk instance of the graph first
        self.create_random_walk_graph()

        if labels is None:
            self._all_paths(steps, keep_last)
        elif labels == 'edges':
            self._all_paths_edges(steps)
        elif labels == 'nodes':
            self._all_paths_nodes(steps)
        elif labels == 'edges_nodes':
            self._all_paths_edges_nodes(steps)
        else:
            raise ValueError('labels argument should be one of the following: edges, nodes, edges_nodes, None.')

        if method == 'sampling':
            if verbose:
                print("Use sampling method to get vector representation.")
            if MC is None:
                MC = self.n_samples(steps, delta, eps)
                if verbose:
                    print("Using number of iterations = {} for delta = {} and eps = {}".format(MC, delta, eps))
            start = time.time()
            patterns = self._sampling(steps, MC, prop=prop)
            finish = time.time()
            if verbose:
                print('Spent {} sec to get vector representation via sampling method.'.format(round(finish - start, 2)))
        elif method == 'exact':
            if verbose:
                print("Use exact method to get vector representation.")
            start = time.time()
            patterns = self._exact(steps, labels = labels, prop=prop, verbose=verbose)
            finish = time.time()
            if verbose:
                print('Spent {} sec to get vector representation via exact method.'.format(round(finish - start, 2)))
        else:
            raise ValueError("Wrong method for Graph2Vec.\n You should choose between {} methods".format(', '.join(self.__methods)))


        vector = []
        if verbose:
            print(patterns)
        for path in self.paths[steps]:
            vector.append(patterns.get(tuple(path), 0))
        return vector, {'meta-paths': self.paths[steps]}

class GraphKernel(object):
    '''
    Calculates a kernel matrix.
    It has methods for:
        - reading graphs from the files (read_graphs())
        - calculating embeddings of graphs based on AWE GK
        - calculating kernel matrix for given matrix of 
            embeddings (so you can provide you own embeddings 
            matrix E by self.embeddings = E)
        - saving/loading embeddings/kernel matrix to the file in compressed format          
    '''
    def __init__(self, graphs = None):
        self.gv = Graph2Vec()
        self.graphs = graphs
        self.__methods = ['dot', 'rbf', 'poly']

    def kernel_value(self, v1, v2, method = 'dot', sigma = 1, c = 0, d = 1):
        '''Calculates kernel value between two vectors.
        methods can be dot, rbf. '''
        if method == 'dot':
            return np.array(v1).dot(v2)
        elif method == 'poly':
            return (np.array(v1).dot(v2) + c)**d
        elif method == 'rbf':
            return np.exp(-np.linalg.norm(np.array(v1) - v2) ** 2 / sigma)
        else:
            raise ValueError("Wrong method for Graph Kernel.\n You should choose between {} methods".format(', '.join(self.__methods)))

    def read_graphs(self, filenames = None, folder = None, ext = None, header=True, weights=True, sep=',', directed=False):
        '''Read graph from the list of files or from the folder.
        If filenames is not None, then read graphs from the list in filenames.
        Then, if folder is not None, then read files from the folder. If extension is specified, then read only files with this extension.
        If folder is not None, then filenames should be named as follows graph0.graphml and should follow the same order as in labels.txt'''
        if filenames is None and folder is None:
            raise ValueError("You should provide list of filenames or folder with graphs")
        if filenames is not None:
            self.graphs = []
            for filename in filenames:
                if filename.split('.')[-1] == 'graphml':
                    G = self.gv.read_graphml(filename)
                else:
                    G = self.gv.read_graph_from_text(filename, header, weights, sep, directed)
                self.graphs.append(G)
        elif folder is not None:
            self.graphs = []
            # have the same order of graphs as in labels.txt
            folder_graphs = filter(lambda g: g.endswith(max(ext, '')), os.listdir(folder))
            sorted_graphs = sorted(folder_graphs, key = lambda g: int(g.split('.')[0][5:]))
            for item in sorted_graphs:
                if ext is not None:
                    if item.split('.')[-1] == ext:
                        if ext == 'graphml':
                            G = self.gv.read_graphml(folder + '/' + item)
                        else:
                            G = self.gv.read_graph_from_text(folder + '/' + item, header, weights, sep, directed)
                        self.graphs.append(G)
                else:
                    if item.split('.')[-1] == 'graphml':
                        G = self.gv.read_graphml(folder + '/' + item)
                    else:
                        G = self.gv.read_graph_from_text(folder + '/' + item, header, weights, sep, directed)
                    self.graphs.append(G)

    def embed_graphs(self, graph2vec_method = 'exact', steps = 3, MC = None, delta = 0.1, eps = 0.1,
                     labels=None, prop=True, keep_last = False):
        if hasattr(self, 'graphs'):
            print('Using {} method to get graph embeddings'.format(graph2vec_method))
            N = len(self.graphs)
            self.gv.graph = self.graphs[0]
            v, d = self.gv.embed(steps = steps, method = graph2vec_method, MC = MC, delta = delta, eps = eps, prop=prop, labels=labels, verbose=False, keep_last = keep_last)
            L = len(v)
            self.embeddings = np.zeros(shape=(N,L))
            self.embeddings[0] = v
            for ix, G in enumerate(self.graphs[1:]):
                if ix % 100 == 0:
                    print('Processing {} graph'.format(ix))
                self.gv.graph = G
                v, d = self.gv.embed(steps = steps, method = graph2vec_method, MC = MC, delta = delta, eps = eps, prop=prop, labels=labels, verbose=False, keep_last = keep_last)
                self.embeddings[ix+1] = v

            # pca = PCA(n_components=0.9)
            # self.embeddings = pca.fit_transform(self.embeddings)
        else:
            raise ValueError('Please, first run read_graphs to create graphs.')

    def kernel_matrix(self, kernel_method = 'rbf', sigma = 1, graph2vec_method = 'exact', steps = 3, MC = None, delta = 0.1, eps = 0.1,
                      prop=True, labels = None, build_embeddings = True, keep_last = False, c=0, d=2):

        if build_embeddings:
            self.embed_graphs(graph2vec_method=graph2vec_method, steps=steps, MC = MC, delta = delta, eps = eps, labels = labels, prop=prop, keep_last=keep_last)

        N = self.embeddings.shape[0]
        self.K = np.zeros(shape=(N,N))

        for i in range(N):
            for j in range(i, N):
                v1 = self.embeddings[i]
                v2 = self.embeddings[j]
                prod = self.kernel_value(v1=v1, v2=v2, method=kernel_method, sigma=sigma, c=c, d=d)
                self.K[i, j] = prod
                self.K[j, i] = prod

    def write_embeddings(self, filename):
        np.savez_compressed(filename, E=self.embeddings)

    def write_kernel_matrix(self, filename):
        np.savez_compressed(filename, K = self.K)

    def load_embeddings(self, filename):
        self.embeddings = np.load(filename)['E']
        return self.embeddings

    def load_kernel_matrix(self, filename):
        self.K = np.load(filename)['K']
        return self.K

class Evaluation(object):
    '''
    Evaluating a Kernel matrix on SVM classification accuracy.
    
    By providing a Kernel matrix M and labels y on initialization, 
    you can run self.evaluate(k=10) to get accuracy results on k=10
    cross validation test sets of your matrix.
    '''
    def __init__(self, matrix, labels, verbose=False):
        '''
        Initialize evaluation.
        :param matrix: feature matrix (either kernel or embeddings)
        :param labels: labels for each row
        '''
        self.M = matrix
        self.y = labels
        self.verbose = verbose

    def split(self, alpha=.8):
        M = self.M
        y = self.y
        K = np.copy(M)
        N, M = K.shape

        perm = np.random.permutation(N)
        for i in range(N):
            K[:, i] = K[perm, i]
        for i in range(N):
            K[i, :] = K[i, perm]

        y = y[perm]

        n1 = int(alpha * N)  # training number
        n2 = int((1 - alpha) / 2 * N)  # validation number

        K_train = K[:n1, :n1]
        y_train = y[:n1]
        K_val = K[n1:(n1 + n2), :n1]
        y_val = y[n1:(n1 + n2)]
        K_test = K[(n1 + n2):, :(n1 + n2)]
        y_test = y[(n1 + n2):]
        K_train_val = K[:(n1 + n2), :(n1 + n2)]
        y_train_val = y[:(n1 + n2)]

        return K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val

    def kfold(self, k=10):
        M = self.M
        y = self.y
        K = np.copy(M)
        N, M = K.shape

        perm = np.random.permutation(N)
        for i in range(N):
            K[:, i] = K[perm, i]
        for i in range(N):
            K[i, :] = K[i, perm]

        y = y[perm]

        test_idx = [(N // k) * ix for ix in range(k)] + [N]
        for ix in range(k):
            test_range = list(range(test_idx[ix], test_idx[ix + 1]))

            train_val_range = [ix for ix in range(N) if ix not in test_range]
            K_train_val = K[np.ix_(train_val_range, train_val_range)]
            y_train_val = y[train_val_range]

            K_test = K[np.ix_(test_range, train_val_range)]
            y_test = y[test_range]

            val_range = random.sample(train_val_range, N // k)
            train_range = [ix for ix in train_val_range if ix not in val_range]
            K_train = K[np.ix_(train_range, train_range)]
            y_train = y[train_range]

            K_val = K[np.ix_(val_range, train_range)]
            y_val = y[val_range]
            yield K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val

    def split_embeddings(self, alpha=.8):
        M = self.M
        y = self.y
        X_train_val, X_test, y_train_val, y_test = train_test_split(M, y, test_size=1 - alpha)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=1 - alpha)
        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val

    def run_SVM(self,
                K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val):
        '''Run SVM on feature matrix (kernel or embeddings) using train-test split.'''
        M, y = self.M, self.y

        C_grid = [0.001, 0.01, 0.1, 1, 10]
        val_scores = []
        for i in range(len(C_grid)):
            # Train a model on Train data
            model = svm.SVC(kernel='precomputed', C=C_grid[i])
            model.fit(K_train, y_train)

            # Predict a model on Validation data
            y_val_pred = model.predict(K_val)
            val_scores.append(accuracy_score(y_val, y_val_pred))

        # re-train a model on Train + Validation data
        max_idx = np.argmax(val_scores)
        model = svm.SVC(kernel='precomputed', C=C_grid[max_idx])
        model.fit(K_train_val, y_train_val)

        # Predict the final model on Test data
        y_test_pred = model.predict(K_test)
        if self.verbose:
            print(y_test_pred)
        return val_scores[max_idx], accuracy_score(y_test, y_test_pred), C_grid[max_idx]

    def evaluate(self, k=10):
        gen = self.kfold(k=k)

        accs = []
        for ix, (K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val) in enumerate(gen):
            val, acc, c_max = self.run_SVM(K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val)
            accs.append(acc)
            if self.verbose:
                print("Scored {} on validation and {} on test with C = {}".format(val, acc, c_max))
        return accs


if __name__ == '__main__':
    np.random.seed(0)

    TRIALS = 10 # number of cross-validation

    STEPS = 2
    KERNEL = 'rbf'
    DATASET = 'imdb_binary'
    METHOD  = 'exact'
    LABELS = None
    PROP = True
    MC = None
    DELTA = 0.1
    EPSILON = 0.1
    C = 0
    D = 1


    parser = argparse.ArgumentParser(description = 'Getting classification accuracy for Graph Kernel Methods')
    parser.add_argument('--dataset', default = DATASET, help = 'Dataset with graphs to classify')
    parser.add_argument('--steps', default = STEPS, help = 'Number of steps for meta-walk', type = int)
    parser.add_argument('--kernel', default = KERNEL, help = 'Kernel type: rbf or dot')

    parser.add_argument('--proportion', default = PROP, help = 'Convert embeddings to be in [0,1]', type = bool)
    parser.add_argument('--labels', default = LABELS, help = 'Labels: edges, nodes, edges_nodes')

    parser.add_argument('--method', default=METHOD, help='Graph2Vec method: sampling or exact')
    parser.add_argument('--MC', default = MC, help = 'Number of times to run random walks for each node', type = int)
    parser.add_argument('--delta', default=DELTA, help='Probability of error to estimate number of samples.', type = float)
    parser.add_argument('--epsilon', default=EPSILON, help='Delta of deviation to estimate number of samples.', type = float)
    parser.add_argument('--C', default=C, help='Free term of polynomial kernel.', type=float)
    parser.add_argument('--D', default=D, help='Power of polynomial kernel.', type=float)

    args = parser.parse_args()

    STEPS = args.steps
    KERNEL = args.kernel
    DATASET = args.dataset
    METHOD = args.method
    LABELS = args.labels
    PROP = args.proportion
    MC = args.MC
    DELTA = args.delta
    EPSILON = args.epsilon
    C = args.C
    D = args.D


    # create a folder for each dataset with output results
    RESULTS_FOLDER = '{}/kernels_v5/'.format(DATASET)
    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    # read labels for each graph
    with open(DATASET + '/labels.txt') as f:
        y = np.array(map(int, f.readlines()))

    # create an instance of a graph kernel and read all graphs
    gk = GraphKernel()
    gk.read_graphs(folder=DATASET, ext='graphml')

    print('Read {} graphs'.format(len(gk.graphs)))
    sys.stdout.flush()

    if KERNEL == 'rbf':
        sigma_grid = [0.0001, 0.001, 0.01, 0.1, 1, 10, 20]
    else:
        sigma_grid = [1]

    for LABELS in [None]:#, 'nodes', 'edges', 'edges_nodes']:
        try:
            for PROP in [True, False]:
                flag = True
                # cross-validation on sigma
                for s_ix in range(len(sigma_grid)):
                    print(DATASET, KERNEL, LABELS, STEPS, PROP, sigma_grid[s_ix])
                    sys.stdout.flush()

                    start2kernelmatrix = time.time()
                    gk.kernel_matrix(kernel_method=KERNEL, graph2vec_method=METHOD, steps=STEPS, prop=PROP, labels=LABELS,
                                     sigma=sigma_grid[s_ix], MC=MC, delta = DELTA, eps = EPSILON,
                                     build_embeddings=flag, keep_last=False, c = C, d = D)
                    finish2kernelmatrix = time.time()
                    print('Time to compute Kernel Matrix: ', finish2kernelmatrix - start2kernelmatrix)
                    sys.stdout.flush()

                    flag = False
                    # write kernel matrix and embeddings
                    # gk.write_kernel_matrix(
                    #     '{}/kernel_{}_{}_{}_{}_{:.2f}_labels.txt'.format(RESULTS_FOLDER, DATASET, KERNEL, LABELS, PROP,
                    #                                                      sigma_grid[s_ix]))
                    # gk.write_embeddings('{}/embeddings_{}_{}_{}_labels.txt'.format(RESULTS_FOLDER, DATASET, LABELS, PROP))

                    N, M = gk.K.shape
                    print('Kernel matrix shape: {}x{}'.format(N, M))
                    sys.stdout.flush()

                    # run SVM with cross-validation on C
                    optimal_val_scores = []
                    optimal_test_scores = []
                    for _ in range(TRIALS):
                        start2SVM = time.time()
                        val, test, C = gk.run_SVM(y, alpha=.9, features='kernels')
                        finish2SVM = time.time()
                        print('{} Time to run SVM: {:.2f}'.format(_, finish2SVM - start2SVM))
                        optimal_val_scores.append(val)
                        optimal_test_scores.append(test)
                        print(val, test, C)

                    print('Average Performance on Validation:', np.mean(optimal_val_scores))
                    print('Average Performance on Test: {:.2f}% +-{:.2f}%'.format(np.mean(optimal_test_scores),
                                                                                  np.std(optimal_test_scores)))
                    sys.stdout.flush()
                    # append results of dataset to the file
                    with open('{}/performance_{}_{}_{}.txt'.format(RESULTS_FOLDER, DATASET, KERNEL, STEPS), 'a') as f:
                        f.write('{} {} {} {} {} {} {} {} {} {}\n'.format(DATASET, KERNEL, LABELS, STEPS, PROP, METHOD, sigma_grid[s_ix],
                                                                   np.mean(optimal_test_scores), np.std(optimal_test_scores),
                                                                      finish2kernelmatrix - start2kernelmatrix))
        except Exception as e:
            print('ERROR FOR', DATASET, KERNEL, LABELS, STEPS, PROP)



    console = []
