import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import itertools
from sklearn.metrics import accuracy_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from Evaluation.DataProcessor import *


N_SPLITS = 10


# CEP, as described in ABC article
def CEP(y_test, preds):
    number_of_misclassified = sum([1 if y != pred else 0 for y, pred in zip(y_test, preds)])
    return -100*(number_of_misclassified/len(preds))


# Check all translations of labels to real labels and returns the best one
def get_best_res(res, reference, fitness=accuracy_score):
    labels = set(reference)
    res_label = set(res)

    num_of_labels_in_res = len(res_label)
    best_res = list(res)
    best_res_fitness = fitness(best_res, reference)
    # Iterate over permutations of real labels
    pers = list(itertools.permutations(labels))
    for per in pers:
        curr_res = list(res)
        curr_replacement_dic = dict(zip(res_label, per[:num_of_labels_in_res]))
        curr_res = [curr_replacement_dic[l] for l in curr_res]
        # for label_to_remove, label_to_insert in curr_replacement_zip:
        #     curr_res[:] = [x if x != label_to_remove else label_to_insert for x in curr_res]
        if fitness(curr_res, reference) > best_res_fitness:
            best_res = curr_res
            best_res_fitness = fitness(best_res, reference)

    return best_res


sns.set_context('poster')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.25, 's' : 80, 'linewidths':0}

dp = DataProcessor(CONFIG_PATH)
dp.multiple_uci_to_csv()
dp.manipulate_multiple_csv()
data = dp.get_multiple_X_y_with_names()

models = [('KMeans', cluster.KMeans), ('MiniBatchKMeans', cluster.MiniBatchKMeans)]

for key in data:
    print("***** {} *****".format(key))
    X, y = data[key][0], data[key][1]

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(X)

    results = []
    names = []

    for name, model in models:

        #***************** Cross Validation *******************#

        # fitness_measures = {'accuracy': 0, 'CEP': 0}
        # kfold = model_selection.KFold(n_splits=N_SPLITS, random_state=7)
        # for train_index, test_index in kfold.split(principalComponents):
        #     # X_train, X_test = X[train_index], X[test_index]
        #     X_train, X_test = principalComponents[train_index], principalComponents[test_index]
        #     y_train, y_test = y[train_index], y[test_index]
        #     m = model(n_clusters=len(list(set(y))))
        #     m.fit(X_train, y_train)
        #     res = m.predict(X_test)
        #     new_res = get_best_res(res, y_test, fitness=CEP)
        #     fitness_measures['accuracy'] += accuracy_score(y_test, new_res)
        #     fitness_measures['CEP'] += CEP(y_test, new_res)
        # for key in ['accuracy', 'CEP']:
        #     fitness_measures[key] = fitness_measures[key]/N_SPLITS
        #
        # t0 = time.time()
        # m_final_preds = m.fit_predict(X)
        # t1 = time.time()
        # fitness_measures['cluster_time'] = int(1000*(t1-t0))
        # print("{}'s accuracy = {}   CEP = {}    cluster_time = {}".format(name,
        #                                                                   fitness_measures['accuracy'],
        #                                                                   -1*fitness_measures['CEP'],
        #                                                                   fitness_measures['cluster_time']))
        # plt.subplot(1, 2, 1)
        # plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y, cmap=plt.cm.Set1,
        #             edgecolor='k')
        # plt.subplot(1, 2, 2)
        # plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=m_final_preds, cmap=plt.cm.Set1,
        #             edgecolor='k')
        #
        # plt.show()

        #******************** Train test split ********************#

        fitness_measures = {'accuracy': 0, 'CEP': 0}
        X_train, X_test, y_train, y_test = train_test_split(principalComponents, y, test_size=0.25, stratify=y)
        m = model(n_clusters=len(list(set(y))))
        t0 = time.time()
        m.fit(X_train, y_train)
        t1 = time.time()
        res = m.predict(X_test)
        t2 = time.time()
        new_res = get_best_res(res, y_test, fitness=CEP)
        fitness_measures['accuracy'] += accuracy_score(y_test, new_res)
        fitness_measures['CEP'] += CEP(y_test, new_res)
        # t0 = time.time()
        # m_final_preds = m.fit_predict(X)
        # t1 = time.time()
        fitness_measures['fit_time'] = int(1000 * (t1 - t0))
        fitness_measures['cluster_time'] = int(1000 * (t2 - t1))
        print("{}'s accuracy = {}   CEP = {}    cluster_time = {}   fit_time = {}"
              .format(name,
                      fitness_measures['accuracy'],
                      -1 * fitness_measures['CEP'],
                      fitness_measures['cluster_time'],
                      fitness_measures['fit_time']))
        plt.subplot(1, 2, 1)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.cm.Set1,
                    edgecolor='k')
        plt.subplot(1, 2, 2)
        plt.scatter(X_test[:, 0], X_test[:, 1], c=res, cmap=plt.cm.Set1,
                    edgecolor='k')

        plt.show()