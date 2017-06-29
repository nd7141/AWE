import networkx as nx
import random, time, math
import numpy as np

class Graph2Vec(object):
    def __init__(self, G = None, RW = None):
        self.graph = G
        self.rw_graph = RW
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

    def create_random_walk_graph(self):
        '''Creates a probabilistic graph from graph with weights.'''
        if self.graph is None:
            raise ValueError, "You should first create a weighted graph."

        RW = nx.DiGraph()
        for node in self.graph:
            edges = self.graph[node]
            total = float(sum([edges[v]['weight'] for v in edges]))
            for v in edges:
                RW.add_edge(node, v, {'weight': edges[v]['weight'] / total})
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

    def _exact(self, steps):
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
        print 'Total walks of size {} in a graph:'.format(steps), len(all_walks)
        return walks

    def run(self, method = 'sampling', steps = None, M = None, delta = 0.1, eps = 0.1):
        '''Generic function to get vector representation.
        method can be sampling, exact
        steps is the number of steps.
        M is the number of iterations.
        delta is probability devitation from the true distribution of meta-walks
        eps is absolute value for deviation of first norm
        Return vector and meta information as dictionary.'''
        if self.rw_graph is None:
            self.create_random_walk_graph()

        if steps is None:
            steps = 5
            print("Use default number of steps = {}".format(steps))

        self._all_paths(steps)

        if method == 'sampling':
            print("Use sampling method to get vector representation.")
            if M is None:
                M = self.n_samples(steps, delta, eps)
                print("Use number of iterations = {} for delta = {} and eps = {}".format(M, delta, eps))
            start = time.time()
            patterns = self._sampling(steps, M)
            finish = time.time()
            print('Spent {} sec to get vector representation via sampling method.'.format(round(finish - start, 2)))
        elif method == 'exact':
            print("Use exact method to get vector representation.")
            start = time.time()
            patterns = self._exact(steps)
            finish = time.time()
            print('Spent {} sec to get vector representation via exact method.'.format(round(finish - start, 2)))
        else:
            raise ValueError, \
                "Wrong method for Graph2Vec.\n You should choose between {} methods".format(', '.join(self.__methods))


        vector = []
        print patterns
        for path in self.paths[steps]:
            vector.append(patterns.get(tuple(path), 0))
        return vector, {'meta-paths':self.paths[steps]}

class GraphKernel(object):
    def __init__(self):
        self.gv = Graph2Vec()
        self.__methods = ['dot', 'rbf']

    def run(self, G1, G2, method = 'dot', sigma = 1, graph2vec_method = 'exact', steps = 3):
        self.gv.graph = G1
        v1, d1 = self.gv.run(graph2vec_method, steps = steps)
        self.gv.graph = G2
        v2, d2 = self.gv.run(graph2vec_method, steps = steps)
        if method == 'dot':
            return np.array(v1).dot(v2)
        elif method == 'rbf':
            return np.exp(-np.linalg.norm(np.array(v1) - v2) ** 2 / sigma)
        else:
            raise ValueError, \
                "Wrong method for Graph Kernel.\n You should choose between {} methods".format(', '.join(self.__methods))


if __name__ == '__main__':
    filename = 'test_graph_original.graphml'
    STEPS = 3
    M = 100

    G = nx.read_graphml(filename)
    gv = Graph2Vec(G = G)
    print gv.run(method = 'exact', steps=STEPS)

    gk = GraphKernel()
    print gk.run(G, G)
    print gk.run(G, G, method='rbf')
