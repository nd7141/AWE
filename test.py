import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random, time, os
import numpy as np
from pprint import pprint
from main import Graph2Vec, GraphKernel


ROOT = '../'
ds = ['imdb_b', 'imdb_m', 'collab', 'reddit_b', 'reddit_m5K', 'reddit_m10k']
#
folder = ROOT + ds[0] + '/'
# ext = 'graphml'
#
# folder_graphs = filter(lambda g: g.endswith(max(ext, '')), os.listdir(folder))
# sorted_graphs = list(enumerate(sorted(folder_graphs, key = lambda g: int(g.split('.')[0][5:]))))
#
# print(sorted_graphs)
# print(random.shuffle(sorted_graphs))
# print(sorted_graphs)

# E = np.loadtxt('embeddings/imdb_b/embeddings.txt')
# # read labels for each graph
# with open(folder + '/labels.txt') as f:
#     y = np.array(list(map(int, f.readlines())))

# gk = GraphKernel()
# gk.embeddings = E
# gk.kernel_matrix(build_embeddings=False)
# val, test, C = gk.run_SVM(y)
# print(val)
# print(test)
# print(C)


g2v = Graph2Vec()
steps = 7
g2v._all_paths(steps, keep_last=True)
print(len(list(g2v.paths[steps])))





console = []