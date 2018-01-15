import matplotlib.pyplot as plt
import networkx as nx
from networkx.algorithms import bipartite
import random, time, os
import numpy as np
from pprint import pprint
from main import Graph2Vec, GraphKernel, Evaluation
from doc2vec import Doc2Vec
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

ROOT = '../'
ds = ['imdb_b', 'imdb_m', 'collab', 'reddit_b', 'reddit_m5K', 'reddit_m10k']
#
folder = ROOT + ds[0] + '/'



DATASET = 'imdb_m'

with open(ROOT + DATASET + '/labels.txt') as f:
    y = np.array(list(map(int, f.readlines())))

E = np.load('doc2vec_results/'+ DATASET + '/embeddings.txt.npz')['E']
# K = np.load('doc2vec_results/'+ DATASET + '/kernel_rbf_20.txt.npz')['K']
ev = Evaluation(E, y, verbose=True)
# accs = ev.evaluate(10)
# print()

accs = []
for _ in range(10):
    X_train, X_val, X_test, y_train, y_val, y_test, X_train_val, y_train_val = ev.split_embeddings(0.8)

    model = svm.SVC(kernel='rbf', C=1)
    model.fit(X_train_val, y_train_val)

    # Predict the final model on Test data
    y_test_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_test_pred)
    accs.append(acc)
    print(y_test_pred)
    print('Accuracy: {}'.format(acc))
print('Mean Accuracy: {}'.format(np.mean(accs)))




console = []