import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random, time, os
import numpy as np
from pprint import pprint
from main import Graph2Vec, GraphKernel
from doc2vec import Doc2Vec

ROOT = '../'
ds = ['imdb_b', 'imdb_m', 'collab', 'reddit_b', 'reddit_m5K', 'reddit_m10k']
#
folder = ROOT + ds[0] + '/'


steps = 7
window_size = 5
dataset = 'reddit_m5k'
folder = '../Datasets/'

d2v = Doc2Vec(dataset = dataset, root=folder, steps=steps, window_size=window_size)
d2v.g2v.read_graphml(d2v.folder + d2v.sorted_graphs[0])
d2v.g2v.create_random_walk_graph()
print('N = {}'.format(len(d2v.g2v.rw_graph)))
s2s = time.time()
d2v.g2v.generate_graph_batch(d2v.window_size, d2v.steps, d2v.walk_ids, 0)
print('Time: {}'.format(time.time() - s2s))

s2s = time.time()
d2v.g2v.generate_random_batch_pvdm(1593, d2v.window_size, d2v.steps, d2v.walk_ids, 0)
print('Time: {}'.format(time.time() - s2s))





console = []