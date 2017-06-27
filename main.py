import networkx as nx
import random, time

class Graph2Vec(object):
    def __init__(self, G = None, RW = None):
        self.graph = G
        self.rw_graph = RW
        self.paths = dict()

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

    # TODO: Optimize function by not running multiple walks for the same node
    def random_walks(self, steps, M, prop=True):
        walks = dict()
        N = len(self.rw_graph)
        for node in self.rw_graph:
            for it in range(M):
                for s in range(2, steps + 1):
                    w = self._random_walk_node(node, s)
                    amount = 1.
                    if prop:
                        amount /= (N * M)
                    if w not in walks:
                        walks[w] = amount
                    else:
                        walks[w] += amount
        return walks

    # TODO: Replace _sampling function with content of random_walk function
    def _sampling(self, steps, M):
        '''Find vector represesntation using sampling method.
        steps is the number of steps.
        M is the number of iterations.
        Returns dictionary pattern to probability.'''
        patterns = dict()
        for _ in range(M):
            u = random.choice(self.rw_graph.nodes())
            walk = self._random_walk_node(u, steps)
            patterns[walk] = patterns.get(walk, 0) + 1. / M
        return patterns

    def run(self, method = 'sampling', steps = None, M = None):
        '''Generic function to get vector representation.
        steps is the number of steps.
        M is the number of iterations.
        Return vector and meta information as dictionary.'''
        if self.rw_graph is None:
            self.create_random_walk_graph()

        if method == 'sampling':
            print("Use sampling method to get vector representation.")
            if steps is None:
                steps = 5
                print("Use default number of steps = {}".format(steps))
            if M is None:
                M = 500
                print("Use default number of iterations = {}".format(M))
            start = time.time()
            patterns = self.random_walks(steps, M)
            finish = time.time()
            print('Spent {} sec to get vector representation via sampling method.'.format(round(finish - start, 2)))

            self._all_paths(steps)
            vector = []
            print patterns
            for path in self.paths[steps]:
                vector.append(patterns.get(tuple(path), 0))
            return vector, {'meta-paths':self.paths[steps]}




if __name__ == '__main__':
    filename = 'test_graph_original.graphml'
    STEPS = 3
    M = 100

    G = nx.read_graphml(filename)
    gv = Graph2Vec(G = G)
    print gv.run(steps = STEPS, M = M)

