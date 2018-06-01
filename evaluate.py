
import os
import math
import random
import argparse
import sys
import time
import re

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from collections import Counter

import numpy as np
import tensorflow as tf
import re

from AnonymousWalkKernel import AnonymousWalks, GraphKernel, Evaluation

from sklearn.model_selection import train_test_split
import json

'''
Evaluate model on embeddings
'''

if __name__ == '__main__':

    # Set random seeds
    SEED = 2018
    random.seed(SEED)
    np.random.seed(SEED)

    dataset = 'mutag'
    RESULTS_FOLDER = 'doc2vec_results/'
    TRIALS = 10  # number of cross-validation
    root = '../Datasets/'
    check_intervals = 60

    parser = argparse.ArgumentParser(description='Getting classification accuracy for Graph Kernel Methods')

    parser.add_argument('--dataset', default=dataset, help='Dataset with graphs to classify')
    parser.add_argument('--results_folder', default=RESULTS_FOLDER, help='Folder to store results')
    parser.add_argument('--root', default=root, help='Root folder of dataset')

    args = parser.parse_args()

    dataset = args.dataset
    RESULTS_FOLDER = args.results_folder
    root = args.root

    if not os.path.exists(RESULTS_FOLDER):
        os.makedirs(RESULTS_FOLDER)

    if not os.path.exists(RESULTS_FOLDER + '/' + dataset):
        os.makedirs(RESULTS_FOLDER + '/' + dataset)

    print('Start evaluating')
    print('DATASET: {}'.format(dataset))
    print('')

    # read classes for each graph
    y = []
    with open(root + dataset + '/labels.txt') as f:
        for line in f:
            y.extend(list(map(int, line.strip().split())))
    y = np.array(y)

    with open('{}/{}/perf_all_{}.txt'.format(RESULTS_FOLDER, dataset, dataset),
              'a') as f:
        f.write('Dataset || Epoch || Kernel || Sigma || Mean || Std || Time\n')

    # read embeddings
    gk = GraphKernel()
    all_files = os.listdir(RESULTS_FOLDER + '/' + dataset + '/' + 'tmp/')

    counters = 0
    while counters < 3600: # when no new embeddings file appears within 24 hours
        current_files = set(os.listdir(RESULTS_FOLDER + '/' + dataset + '/' + 'tmp/'))
        new_file = list(current_files.difference(all_files))
        if len(new_file) > 0: # new file appeared
            print('Found new embedding file {}'.format(new_file[0]))
            embedding_file = new_file[0]
            epoch = re.findall('\d+', embedding_file)[0]
            all_files = current_files
            counters = 0

            gk.embeddings = gk.load_embeddings(RESULTS_FOLDER + '/' + dataset + '/tmp/{}'.format(embedding_file))
            ### testing on embeddings
            for _ in range(3):
                E = gk.embeddings
                idx_train, idx_test = train_test_split(list(range(E.shape[0])), test_size=0.2)
                E_train = E[idx_train, :]
                y_train = y[idx_train]
                E_test = E[idx_test, :]
                y_test = y[idx_test]

                model = svm.SVC(kernel='rbf', C=1)
                model.fit(E_train, y_train)
                y_predicted = model.predict(E_test)
                print('On Embeddings:', accuracy_score(y_test, y_predicted))

            ################## Estimate results: Classification Accuracy ########################
            print()
            for KERNEL in ['rbf', 'linear', 'poly']:

                if KERNEL == 'rbf':
                    sigma_grid = [0.1, 1, 10]
                else:
                    sigma_grid = [1]

                # try:
                for s_ix in range(len(sigma_grid)):
                    print('Setup: ', dataset, KERNEL, sigma_grid[s_ix])
                    sys.stdout.flush()

                    print('Computing Kernel Matrix...')
                    start2kernelmatrix = time.time()
                    gk.kernel_matrix(kernel_method=KERNEL, build_embeddings=False, sigma=sigma_grid[s_ix])
                    finish2kernelmatrix = time.time()
                    print('Time to compute Kernel Matrix: ', finish2kernelmatrix - start2kernelmatrix)
                    sys.stdout.flush()

                    N, M = gk.K.shape
                    print('Kernel matrix shape: {}x{}'.format(N, M))
                    sys.stdout.flush()

                    # run k-fold SVM with cross-validation on C
                    print('Evaluating Kernel Matrix on SVM...')
                    ev = Evaluation(gk.K, y, verbose=False)
                    optimal_test_scores = []
                    for _ in range(TRIALS):
                        print(TRIALS - _, end=' ')
                        sys.stdout.flush()
                        accs = ev.evaluate()
                        optimal_test_scores.extend(accs)
                    print()
                    print('Average Performance on Test: {:.2f}% +-{:.2f}%'.format(100 * np.mean(optimal_test_scores),
                                                                                  100 * np.std(optimal_test_scores)))
                    sys.stdout.flush()
                    # append results of dataset to the file
                    with open('{}/{}/perf_all_{}.txt'.format(RESULTS_FOLDER, dataset, dataset),
                              'a') as f:
                        f.write('{} {} {} {} {} {} {}\n'.format(dataset, epoch, KERNEL, sigma_grid[s_ix],
                                                             np.mean(optimal_test_scores), np.std(optimal_test_scores),
                                                             finish2kernelmatrix - start2kernelmatrix))
                    print()

        else:
            time.sleep(check_intervals)
            counters += 1