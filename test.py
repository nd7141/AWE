import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random, time
import numpy as np
from pprint import pprint
from main import Graph2Vec, GraphKernel


################### Preparing Features: counts of labels

gk = GraphKernel()
gk.read_graphs(folder = 'bio/{}'.format('mutag'))

features = np.zeros(shape = (len(gk.graphs), 2))
for ix, G in enumerate(gk.graphs):
    features[ix, 0] = len(G)
    features[ix, 1] = len(G.edges())


################### Get labels
with open('bio/mutag_label.txt') as f:
    y = np.array(map(int, f.readlines()[0].split()))

################### Get Kernel and Run Prediction
gk.embeddings = features
gk.kernel_matrix(build_embeddings=False, kernel_method='dot')

print gk.run_SVM(y, features='kernels')

console = []