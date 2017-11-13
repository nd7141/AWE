import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random, time, os
import numpy as np
from pprint import pprint
from main import Graph2Vec, GraphKernel
import collections
from main import *

def read_matrix(filename):
    with open(filename) as f:
        for en, line in enumerate(f):
            row = list(map(float, line.split(',')[:-1]))
            if en == 0:
                N = len(row)
                K = np.zeros((N, N))
            K[en, :] = row
    return K



if __name__ == '__main__':

    DATASET = 'ib'
    path_to_datasets = '../Datasets/'
    TRIALS = 1
    kernel_path = 'imdb_b_kR.kernel'

    labels_file = path_to_datasets + DATASET + '/labels.txt'
    with open(labels_file) as f:
        y = np.array(list(map(int, f.readlines())))


    K = read_matrix(kernel_path)

    ev = Evaluation(K, y, verbose=True)
    # run SVM with cross-validation on C
    optimal_test_scores = []
    for _ in range(TRIALS):
        print(_)
        accs = ev.evaluate()
        optimal_test_scores.extend(accs)
    print('Average Performance on Test: {:.2f}% +-{:.2f}%'.format(100 * np.mean(optimal_test_scores),
                                                                  100 * np.std(optimal_test_scores)))

    del ev  # preventing memory leak

    console = []