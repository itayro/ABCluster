import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection
from sklearn.decomposition import PCA
import itertools
from sklearn.metrics import accuracy_score
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from Evaluation.DataProcessor import *


N_SPLITS = 10

# Check all translations of labels to real labels and returns the best one
def get_best_res(res, reference, fitness=accuracy_score):
    labels = set(reference)
    res_label = set(res)

    num_of_labels_in_res = len(res_label)
    best_res = list(res)
    best_res_fitness = accuracy_score(best_res, reference)
    # Iterate over permutations of real labels
    for per in list(itertools.permutations(labels)):
        curr_res = list(res)
        for label_to_remove, label_to_insert in zip(res_label, per[:num_of_labels_in_res]):
            curr_res[:] = [x if x != label_to_remove else label_to_insert for x in curr_res]
        if accuracy_score(curr_res, reference) > best_res_fitness:
            best_res = curr_res
            best_res_fitness = accuracy_score(best_res, reference)

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

    # TODO: Visualize using PCA (above) & find good scoring method
    for name, model in models:
        kfold = model_selection.KFold(n_splits=N_SPLITS, random_state=7)
        fitness_measures = {'accuracy': 0}
        for train_index, test_index in kfold.split(principalComponents):
            X_train, X_test = principalComponents[train_index], principalComponents[test_index]
            y_train, y_test = y[train_index], y[test_index]
            m = model(n_clusters=len(list(set(y))))
            m.fit(X_train, y_train)
            res = m.predict(X_test)
            new_res = get_best_res(res, y_test)
            fitness_measures['accuracy'] += accuracy_score(y_test, new_res)
        for key in fitness_measures:
            fitness_measures[key] = fitness_measures[key]/N_SPLITS
        print("{}'s accuracy = {}".format(name, fitness_measures['accuracy']))

        plt.subplot(1, 2, 1)
        plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=y, cmap=plt.cm.Set1,
                    edgecolor='k')
        plt.subplot(1, 2, 2)
        plt.scatter(principalComponents[:, 0], principalComponents[:, 1], c=m.fit_predict(X), cmap=plt.cm.Set1,
                    edgecolor='k')

        plt.show()

        # cv_results = model_selection.cross_val_score(model(n_clusters=len(list(set(y)))), X, y, cv=kfold, scoring='accuracy')
        # #cv_results = model_selection.cross_val_score(model(), X, y, cv=kfold, scoring='homogeneity_score')
        # results.append(cv_results)
        # names.append(name)
        # msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        # print(msg)