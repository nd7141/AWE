import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random, time
import numpy as np
from pprint import pprint
from main import Graph2Vec, GraphKernel


################### Preparing Features: counts of labels
def count_labels(attrs):
    d = dict()
    for node in attrs:
        d[attrs[node]] = d.get(attrs[node], 0) + 1
    return d

gk = GraphKernel()
gk.read_graphs(folder = 'bio/{}'.format('mutag'))

G = gk.graphs[0]

counts_nodes = []
counts_edges = []
for G in gk.graphs:
    counts_nodes.append(count_labels(nx.get_node_attributes(G, 'label')))
    counts_edges.append(count_labels(nx.get_edge_attributes(G, 'label')))

node_labels = set()
edge_labels = set()
for ix in range(len(counts_nodes)):
    node_labels.update(counts_nodes[ix].keys())
    edge_labels.update(counts_edges[ix].keys())

features = np.zeros(shape = (len(counts_nodes), len(node_labels) + len(edge_labels)))
order = {el: ix for ix, el in enumerate(list(node_labels) + list(edge_labels))}

for ix, counts in enumerate(counts_nodes):
    for el in counts:
        features[ix, order[el]] = counts[el]

for ix, counts in enumerate(counts_edges):
    for el in counts:
        features[ix, order[el]] = counts[el]

################### Get labels
with open('bio/mutag_label.txt') as f:
    y = np.array(map(int, f.readlines()[0].split()))

################### Get Kernel and Run Prediction
gk.embeddings = features
gk.kernel_matrix(build_embeddings=False, kernel_method='rbf')

print gk.run_SVM(y, features='phimap')

console = []