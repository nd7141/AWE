import networkx as nx
import random, time, math, os
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

class Graph2Vec(object):
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
            raise ValueError, "You should first create a weighted graph."

        # get name of the label on graph edges (assume all label names are the same)
        label_name = 'weight'
        # for e in self.graph.edges_iter(data=True):
        #     label_name = e[2].keys()[0]
        #     break

        RW = nx.DiGraph()
        for node in self.graph:
            edges = self.graph[node]
            total = float(sum([edges[v].get(label_name, 1) for v in edges]))
            for v in edges:
                RW.add_edge(node, v, {'weight': edges[v].get(label_name,1) / total})
        self.rw_graph = RW

    def _all_paths(self, steps):
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
            raise ValueError, "Create a Random Walk graph first with {}".format(self.create_random_walk_graph.__name__)
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

    def _sampling(self, steps, M, prop=True):
        '''Find vector representation using sampling method.
        Run M random walks with n steps for each node in the graph.
        steps is the number of steps.
        M is the number of iterations.
        Returns dictionary pattern to probability.
        '''
        walks = dict()
        N = len(self.rw_graph)
        for node in self.rw_graph:
            for it in range(M):
                # run a random walk with n steps, and then look at sub-walks
                w = self._random_walk_node(node, steps)
                for length in range(3, len(w) + 1):
                    w_cropped = w[:length]
                    amount = 1.
                    if prop:
                        amount /= (N * M)
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
                    raise ValueError, 'labels argument should be one of the following: edges, nodes, edges_nodes, None.'
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

    def embed(self, method = 'exact', steps = None, M = None, delta = 0.1, eps = 0.1, prop=True, labels = None, verbose = True):
        '''Generic function to get vector representation.
        method can be sampling, exact
        steps is the number of steps.
        M is the number of iterations.
        labels, possible values None (no labels), 'edges', 'nodes', 'edges_nodes'.
        delta is probability devitation from the true distribution of meta-walks
        eps is absolute value for deviation of first norm
        Return vector and meta information as dictionary.'''

        # Create a random walk instance of the graph first
        self.create_random_walk_graph()

        if steps is None:
            steps = 5
            if verbose:
                print("Use default number of steps = {}".format(steps))

        if labels is None:
            self._all_paths(steps)
        elif labels == 'edges':
            self._all_paths_edges(steps)
        elif labels == 'nodes':
            self._all_paths_nodes(steps)
        elif labels == 'edges_nodes':
            self._all_paths_edges_nodes(steps)
        else:
            raise ValueError, 'labels argument should be one of the following: edges, nodes, edges_nodes, None.'

        if method == 'sampling':
            if verbose:
                print("Use sampling method to get vector representation.")
            if M is None:
                M = self.n_samples(steps, delta, eps)
                if verbose:
                    print("Use number of iterations = {} for delta = {} and eps = {}".format(M, delta, eps))
            start = time.time()
            patterns = self._sampling(steps, M, prop=prop)
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
            raise ValueError, \
                "Wrong method for Graph2Vec.\n You should choose between {} methods".format(', '.join(self.__methods))


        vector = []
        if verbose:
            print patterns
        for path in self.paths[steps]:
            vector.append(patterns.get(tuple(path), 0))
        return vector, {'meta-paths': self.paths[steps]}

class GraphKernel(object):
    def __init__(self, graphs = None):
        self.gv = Graph2Vec()
        self.graphs = graphs
        self.__methods = ['dot', 'rbf']

    def kernel_value(self, v1, v2, method = 'dot', sigma = 1):
        '''Calculates kernel value between two vectors.
        methods can be dot, rbf. '''
        if method == 'dot':
            return np.array(v1).dot(v2)
        elif method == 'rbf':
            return np.exp(-np.linalg.norm(np.array(v1) - v2) ** 2 / sigma)
        else:
            raise ValueError, \
                "Wrong method for Graph Kernel.\n You should choose between {} methods".format(', '.join(self.__methods))

    def read_graphs(self, filenames = None, folder = None, ext = None, header=True, weights=True, sep=',', directed=False):
        '''Read graph from the list of files or from the folder.
        If filenames is not None, then read graphs from the list in filenames.
        Then, if folder is not None, then read files from the folder. If extension is specified, then read only files with this extension.'''
        if filenames is None and folder is None:
            raise ValueError, "You should provide list of filenames or folder with graphs"
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
            for item in os.listdir(folder):
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

    def embed_graphs(self, graph2vec_method = 'exact', steps = 3, M = None, delta = 0.1, eps = 0.1, labels=None, prop=True):
        if hasattr(self, 'graphs'):
            print('Using {} method to get graph embeddings'.format(graph2vec_method))
            N = len(self.graphs)
            self.gv.graph = self.graphs[0]
            v, d = self.gv.embed(graph2vec_method, steps, M = M, delta = delta, eps = eps, prop=prop, labels=labels, verbose=False)
            M = len(v)
            self.embeddings = np.zeros(shape=(N,M))
            self.embeddings[0] = v
            for ix, G in enumerate(self.graphs[1:]):
                self.gv.graph = G
                v, d = self.gv.embed(graph2vec_method, steps, M = M, delta = delta, eps = eps, prop=prop, labels=labels, verbose=False)
                self.embeddings[ix+1] = v
            self.meta = d

            # pca = PCA(n_components=0.9)
            # self.embeddings = pca.fit_transform(self.embeddings)
        else:
            raise ValueError, 'Please, first run read_graphs to create graphs.'

    def kernel_matrix(self, kernel_method = 'rbf', sigma = 1, graph2vec_method = 'exact', steps = 3, M = None, delta = 0.1, eps = 0.1, prop=True, labels = None):

        self.embed_graphs(graph2vec_method, steps, M = M, delta = delta, eps = eps, labels = labels, prop=prop)

        N = len(self.graphs)
        self.K = np.zeros(shape=(N,N))

        for i in range(N):
            for j in range(i, N):
                v1 = self.embeddings[i]
                v2 = self.embeddings[j]
                prod = self.kernel_value(v1=v1, v2=v2, method=kernel_method, sigma=sigma)
                self.K[i, j] = prod
                self.K[j, i] = prod

    def write_embeddings(self, filename):
        np.savetxt(filename, self.embeddings, fmt='%.3f')

    def write_kernel_matrix(self, filename):
        np.savetxt(filename, self.K, fmt='%.3f')

    def split(self, y, alpha = .8):
        K = np.copy(self.K)
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
        K_test = K[(n1 + n2):, :n1]
        y_test = y[(n1 + n2):]
        K_train_val = K[:(n1 + n2), :(n1+n2)]
        y_train_val = y[:(n1 + n2)]

        return K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val

    def split_embeddings(self, y, alpha = .8):
        X_train_val, X_test, y_train_val, y_test = train_test_split(self.embeddings, y, test_size = 1 - alpha )
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size = 1 - alpha)
        return X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val

    def run_SVM(self, y, alpha = .8, lower = 10**(-3), upper = 10, num = 10):
        K_train, K_val, K_test, y_train, y_val, y_test, K_train_val, y_train_val = self.split(y, alpha)

        # C_grid = np.linspace(lower, upper, num=num)
        C_grid = 10.**(-np.arange(-1, 11))
        val_scores = []
        test_scores = []
        for i in range(len(C_grid)):
            model = svm.SVC(kernel='precomputed', C=C_grid[i])
            # model = svm.SVC(C=C_grid[i])
            model.fit(K_train, y_train)

            y_val_pred = model.predict(K_val)
            val_scores.append(accuracy_score(y_val, y_val_pred))

            y_test_pred = model.predict(K_test)
            test_scores.append(accuracy_score(y_test, y_test_pred))

        print 'Last prediction values: ', y_val_pred
        max_idx = np.argmax(val_scores)
        return val_scores[max_idx], test_scores[max_idx], C_grid[max_idx]

        # model = svm.SVC(kernel = 'precomputed', C = C_grid[max_idx])
        # # model = svm.SVC(C=C_grid[max_idx])
        # model.fit(K_train_val, y_train_val)
        #
        # y_test_pred = model.predict(K_test)
        # print y_test_pred
        # return val_scores[max_idx], accuracy_score(y_test, y_test_pred), C_grid[max_idx]


if __name__ == '__main__':
    STEPS = 2
    M = 10
    TRIALS = 10
    KERNEL = 'dot'
    DATASET = 'mutag'
    RESULTS_FOLDER = 'test_{}/'.format(KERNEL)
    LABELS = None

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    gk = GraphKernel()
    # gk.read_graphs(filenames = ['bio/mutag/mutag_1.graphml', 'bio/mutag/mutag_188.graphml'], directed = True)
    gk.read_graphs(folder = 'bio/{}'.format(DATASET))
    # gk.kernel_matrix(kernel_method=KERNEL, steps = STEPS)
    # print gk.embeddings, gk.meta


    # gk.write_kernel_matrix('{}/kernel_{}_{}_{}_labels.txt'.format(RESULTS_FOLDER, DATASET, KERNEL, LABELS))
    # gk.write_embeddings('{}/embeddings_{}_{}_labels.txt'.format(RESULTS_FOLDER, DATASET, LABELS))


    # with open('bio/' + DATASET + '_label.txt') as f:
    #     y = np.array(map(int, f.readlines()[0].split()))
    #
    # K = np.loadtxt('mutag_no_labels_ker_mat.txt')
    # print K.shape
    # K = np.loadtxt('kernels_rbf/mutag_kernel_wl.txt')
    # print K.shape
    #
    #
    # sigma_grid = np.linspace(10**(-4), 10, num=10)
    # sigma_grid = 10.**(-np.arange(1, 11))
    # sigma_grid = [1]
    # sigma_test_score = []
    # for six in range(len(sigma_grid)):
    #     print 'Sigma:', sigma_grid[six]
    #     gk.kernel_matrix(kernel_method=KERNEL, steps=STEPS, prop=False, sigma = sigma_grid[six])
    #     gk.K = K
    #     tests = []
    #     for _ in range(TRIALS):
    #         results = gk.run_SVM(y, alpha = .8)
    #         tests.append(results[1])
    #     print 'Result: ', np.mean(tests)
    #     sigma_test_score.append(np.mean(tests))
    #
    # max_idx = np.argmax(sigma_test_score)
    # print sigma_test_score[max_idx]
    # print sigma_grid[max_idx]
    # print sigma_test_score




    # plt.figure(figsize=(15, 8))
    # for i, G in enumerate(gk.graphs):
    #     plt.subplot(1, 2, i+1)
    #     pos = nx.shell_layout(G)
    #     nx.draw_networkx(G, pos = pos, with_labels = True)
    #     plt.title('Graph {}'.format((189 - i) % 189))
    # plt.show()


    for DATASET in ['mutag']: #, 'enzymes', 'DD', 'NCI1', 'NCI109']:

        with open('bio/' + DATASET + '_label.txt') as f:
            y = np.array(map(int, f.readlines()[0].split()))

        gk = GraphKernel()
        gk.read_graphs(folder = 'bio/' + DATASET)

        #TODO: adapt algorithm to consider labels
        for LABELS in [None]: #, 'edges', 'nodes', 'edges_nodes']:
            try:
                # gk.kernel_matrix(KERNEL, steps=STEPS, prop=False, labels=LABELS)
                #
                # K = gk.K
                # gk.write_kernel_matrix('{}/kernel_{}_{}_{}_labels.txt'.format(RESULTS_FOLDER, DATASET, KERNEL, LABELS))
                # gk.write_embeddings('{}/embeddings_{}_{}_labels.txt'.format(RESULTS_FOLDER, DATASET, LABELS))

                K = np.loadtxt('mutag_wl_ker_mat.txt')
                gk.K = K

                N, M = K.shape
                print 'Kernel matrix shape: {}x{}'.format(N, M)

                optimal_val_scores = []
                optimal_test_scores = []
                for _ in range(TRIALS):
                    val, test, C = gk.run_SVM(y, num = 10, alpha = .9)
                    optimal_val_scores.append(val)
                    optimal_test_scores.append(test)
                    print val, test, C

                print 'Average Performance on Validation:', np.mean(optimal_val_scores)
                print 'Average Performance on Test: {:.2f}% +-{:.2f}%'.format(np.mean(optimal_test_scores), np.std(optimal_test_scores))
                # with open('{}/performance.txt'.format(RESULTS_FOLDER), 'a') as f:
                #     f.write('{} {} {} {}\n'.format(DATASET, LABELS, np.mean(optimal_test_scores), np.std(optimal_test_scores)))
            except Exception, e:
                print e
                print 'Exit with error'


    console = []