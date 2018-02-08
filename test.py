import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random, time, os
import numpy as np
from pprint import pprint
from main import Graph2Vec, GraphKernel, Evaluation
# from doc2vec import Doc2Vec
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import math
from collections import Counter
import json

random.seed(2018)
np.random.seed(2018)

ROOT = '../'
ds = ['imdb_b', 'imdb_m', 'collab', 'reddit_b', 'reddit_m5K', 'reddit_m10k']
#
folder = ROOT + ds[0] + '/'



DATASET = 'imdb_b'

# for DATASET in ds[1:]:
#     gk = GraphKernel()
#     gk.read_graphs(folder='../' + DATASET, ext='graphml')
#     gk.embed_graphs(graph2vec_method='sampling', steps = 7, delta = 0.1, eps = 0.5, prop=False, keep_last=True)
#     E = gk.embeddings
#     cnt = Counter()
#     for row in range(E.shape[0]):
#         cnt.update(E[row, :])
#
#     coords = list(cnt.items())
#     s_coords = sorted(coords)
#     with open(DATASET + '_hist.json', 'w') as f:
#         json.dump(s_coords, f)

    # vector = cnt.items()
    # print(vector)

# vector = np.mean(gk.embeddings, axis = 0)
# print(vector)
# G = gk.graphs[1]
# g2v = Graph2Vec()
# g2v.read_graphml(filename = '../' + DATASET + '/graph0.graphml')
# g2v.create_random_walk_graph()
# result = g2v.embed(steps = 7, method = 'sampling', eps = 0.5, delta = 0.1, keep_last=True, prop=False)
# vector = result[0]
# coords = list(Counter(vector).items())


# coords = list(cnt.items())
# s_coords = sorted(coords)
# with open(DATASET + '_hist.json', 'w') as f:
#     json.dump(s_coords, f)
# x, y = list(zip(*s_coords))
# plt.bar(x, y)
# plt.xlim(0, 10)
# plt.show()
# print(vector)

# plt.hist(vector, bins=np.arange(-1, 22, 2)*0.5)
# plt.show()
# for i in range(2, 13):
#     s2t = time.time()
#     g2v._all_paths(steps = i, keep_last=True)
#     print('i = {} Time: {}'.format(i, time.time() - s2t))
#     ls.append(len(g2v.paths[i]))
# print(ls)





a = 15
def f(eps, sigma): print(math.ceil(2./eps**2*(math.log(2**a - 2) - math.log(sigma))))

# f(0.1, 0.05)
# f(0.1, 0.01)

# print(len(g2v.paths[3]))
# print(g2v.embed(steps = 3, method = 'exact', keep_last=True))
# print(g2v._exact(steps = 2, prop=True))


# from networkx import erdos_renyi_graph as erg
#
# folder = 'erdos_renyi_graphs/'
# if not os.path.exists(folder):
#     os.mkdir(folder)
#
# for mu in [2., 3., 4., 5.]:
#     folder2 = folder + 'mu{}/'.format(int(mu))
#     if not os.path.exists(folder2):
#         os.mkdir(folder2)
#     for n in [30000]:
#         folder3 = folder2 + 'n{}/'.format(n)
#         if not os.path.exists(folder3):
#             os.mkdir(folder3)
#
#         p = mu/n
#         print(n, p)
#         for _ in range(10):
#             G = erg(n=n, p=p)
#             G.remove_nodes_from(nx.isolates(G))
#             print(len(G), len(G.edges()))
#             nx.write_graphml(G, folder3 + 'graph{}.graphml'.format(_))
#         print()


# N = 10
# p = 5/N
#
# print(len(G), len(G.edges()))
# nx.draw(G)
# plt.show()
# console = []