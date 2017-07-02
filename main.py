import networkx as nx
import random, time, math, os
import numpy as np

class Graph2Vec(object):
    def __init__(self, G = None):
        self._graph = G
        # paths are dictionary between step and all-paths
        self.paths = dict()
        self.__methods = ['sampling', 'exact']

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

    def read_graphml(self, filename):
        return nx.read_graphml(filename)

    def create_random_walk_graph(self):
        '''Creates a probabilistic graph from graph with weights.'''
        if self.graph is None:
            raise ValueError, "You should first create a weighted graph."

        # get name of the label on graph edges (assume all label names are the same)
        label_name = 'weight'
        for e in self.graph.edges_iter(data=True):
            label_name = e[2].keys()[0]
            break

        RW = nx.DiGraph()
        for node in self.graph:
            edges = self.graph[node]
            total = float(sum([edges[v][label_name] for v in edges]))
            for v in edges:
                RW.add_edge(node, v, {'weight': edges[v][label_name] / total})
        self.rw_graph = RW

    def _all_paths(self, steps):
        '''Get all possible meta-paths of length up to steps.'''
        if self.paths.get(steps) is None:
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
            self.paths[steps] = paths

    def walk2pattern(self, walk):
        '''Converts a walk with arbitrary nodes to meta-walk.'''
        idx = 0
        pattern = []
        d = dict()
        for node in walk:
            if node not in d:
                d[node] = idx
                idx += 1
            pattern.append(d[node])
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

    def _exact(self, steps, verbose = True):
        '''Find vector representation using exact method.
            Calculates probabilities from each node to all other nodes within n steps.
            Running time is the O(# number of random walks) <= O(n*d_max^steps).
            steps is the number of steps.
            Returns dictionary pattern to probability.
        '''
        walks = dict()
        all_walks = []

        def patterns(RW, node, steps, walks, current_walk=None, current_dist=1.):
            if current_walk is None:
                current_walk = [node]
            if len(current_walk) > 2:  # walks with more than 1 edge
                all_walks.append(current_walk)
                walks[self.walk2pattern(current_walk)] = walks.get(self.walk2pattern(current_walk), 0) \
                                                         + current_dist / len(RW)
            if steps > 0:
                for v in RW[node]:
                    patterns(RW, v, steps - 1, walks, current_walk + [v], current_dist * RW[node][v]['weight'])

        for node in self.rw_graph:
            patterns(self.rw_graph, node, steps, walks)
        if verbose:
            print('Total walks of size {} in a graph:'.format(steps), len(all_walks))
        return walks

    def embed(self, method = 'sampling', steps = None, M = None, delta = 0.1, eps = 0.1, verbose = True):
        '''Generic function to get vector representation.
        method can be sampling, exact
        steps is the number of steps.
        M is the number of iterations.
        delta is probability devitation from the true distribution of meta-walks
        eps is absolute value for deviation of first norm
        Return vector and meta information as dictionary.'''

        # Create a random walk instance of the graph first
        self.create_random_walk_graph()

        if steps is None:
            steps = 5
            if verbose:
                print("Use default number of steps = {}".format(steps))

        self._all_paths(steps)

        if method == 'sampling':
            if verbose:
                print("Use sampling method to get vector representation.")
            if M is None:
                M = self.n_samples(steps, delta, eps)
                if verbose:
                    print("Use number of iterations = {} for delta = {} and eps = {}".format(M, delta, eps))
            start = time.time()
            patterns = self._sampling(steps, M)
            finish = time.time()
            if verbose:
                print('Spent {} sec to get vector representation via sampling method.'.format(round(finish - start, 2)))
        elif method == 'exact':
            if verbose:
                print("Use exact method to get vector representation.")
            start = time.time()
            patterns = self._exact(steps, verbose=verbose)
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

    def embed_graphs(self, graph2vec_method = 'exact', steps = 3, M = None, delta = 0.1, eps = 0.1):
        if hasattr(self, 'graphs'):
            print('Using {} method to get graph embeddings'.format(graph2vec_method))
            N = len(self.graphs)
            self.gv.graph = self.graphs[0]
            v, d = self.gv.embed(graph2vec_method, steps, M = M, delta = delta, eps = eps, verbose=False)
            M = len(v)
            self.embeddings = np.zeros(shape=(N,M))
            self.embeddings[0] = v
            for ix, G in enumerate(self.graphs[1:]):
                self.gv.graph = G
                v, d = self.gv.embed(graph2vec_method, steps, M = M, delta = delta, eps = eps, verbose=False)
                self.embeddings[ix+1] = v
            self.meta = d
        else:
            raise ValueError, 'Please, first run read_graphs to create graphs.'

    def kernel_matrix(self, kernel_method = 'dot', sigma = 1, graph2vec_method = 'exact', steps = 3, M = None, delta = 0.1, eps = 0.1):

        self.embed_graphs(graph2vec_method, steps, M = M, delta = delta, eps = eps)

        N = len(self.graphs)
        self.K = np.zeros(shape=(N,N))

        for i in range(N):
            for j in range(i, N):
                v1 = self.embeddings[i]
                v2 = self.embeddings[j]
                prod = self.kernel_value(v1=v1, v2=v2, method=kernel_method, sigma=sigma)
                self.K[i, j] = prod
                self.K[j, i] = prod

if __name__ == '__main__':
    filename = 'test_graph_original.graphml'
    STEPS = 3
    M = 100
    dataset = 'bio/mutag'

    with open(dataset + '_label.txt') as f:
        y = np.array(map(int, f.readlines()[0].split()))

    gk = GraphKernel()
    gk.read_graphs(folder = dataset)
    gk.kernel_matrix('rbf', steps = STEPS)

    K = gk.K

    N, M = K.shape
    print 'Kernel matrix shape: {}x{}'.format(N, M)

    # permute input data
    perm = np.random.permutation(N)
    for i in range(N):
        K[:, i] = K[perm, i]
    for i in range(N):
        K[i, :] = K[i, perm]

    y = y[perm]
    print y

    alpha = .5
    n1 = int(alpha*N) # training number
    n2 = int((1-alpha)/2*N) # validation number
    K_train = K[:n1, :n1]
    y_train = y[:n1]
    K_val = K[n1:(n1+n2), :n1]
    y_val = y[n1:(n1+n2)]
    K_test = K[(n1+n2):, :n1]
    y_test = y[(n1+n2):]

    from sklearn import svm
    from sklearn.metrics import accuracy_score

    C_grid = np.linspace(10**-5, 10, num=100)
    val_scores = []
    test_scores = []
    for i in range(len(C_grid)):
        print C_grid[i],
        model = svm.SVC(kernel='precomputed', C = C_grid[i])
        model.fit(K_train, y_train)

        y_val_pred = model.predict(K_val)
        print y_val_pred
        val_scores.append(accuracy_score(y_val, y_val_pred))

        y_test_pred = model.predict(K_test)
        test_scores.append(accuracy_score(y_test, y_test_pred))

    print val_scores
    print test_scores
    # C_idx = np.argmax(val_scores)
    # C_best = C_grid[C_idx]
    # best_test_acc = test_scores[C_idx]

    # from sklearn import datasets
    # from sklearn.model_selection import train_test_split
    #
    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    #
    # N = X.shape[0]
    # K = np.dot(X, X.T)
    # perm = np.random.permutation(N)
    # for i in range(N):
    #     K[:, i] = K[perm, i]
    # for i in range(N):
    #     K[i, :] = K[i, perm]
    #
    # y = y[perm]
    #
    # alpha = .5
    # n1 = int(alpha * N)  # training number
    # n2 = int((1-alpha)/2 * N)  # validation number
    # K_train = K[:n1, :n1]
    # y_train = y[:n1]
    # K_val = K[n1:(n1 + n2), :n1]
    # y_val = y[n1:(n1 + n2)]
    # K_test = K[(n1 + n2):, :n1]
    # y_test = y[(n1 + n2):]
    #
    # C_grid = np.linspace(10**(-5), 10, num=100)
    # for i in range(len(C_grid)):
    #     print C_grid[i],
    #     model = svm.SVC(kernel = 'precomputed', C = C_grid[i])
    #     model.fit(K_train, y_train)
    #
    #     y_test_pred = model.predict(K_test)
    #     print accuracy_score(y_test, y_test_pred)


    console = []