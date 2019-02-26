import matplotlib.pyplot as plt
from ClusteringAlgorithm.ABClustering import ABClustering
from ClusteringAlgorithm.ABClassifier import ABClassifier
from utils.ClusteringObjectiveFunction import *
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import time


"""
      plotting the points where the x & y axises are the second and forth features 
      the left graph illustrates the clusters in the first round and the right graph illustrates
      the clusters in the end of the ABC algorithm
"""


def clustering_with_abc():
    data = MinMaxScaler().fit_transform(load_iris()['data'][:, [1, 3]])
    objective_function1 = SSE(dim=6, n_centroids=3, data=data)
    abc1 = ABClustering(objective_function=objective_function1, colony_size=30, cycles=300, max_tries_employee=100,
                        max_tries_onlooker=100, processing_opt=None)

    abc1.optimize()

    plt.figure(figsize=(9, 8))
    plt.subplot(1, 2, 1)
    centroids = dict(enumerate(decode_centroids(abc1.optimal_value_tracking[0][0],
                                                n_clusters=3, data=data)))

    custom_tgt = []
    for instance in data:
        custom_tgt.append(assign_centroid(centroids, instance))

    colors = ['r', 'g', 'y']

    for instance, tgt in zip(data, custom_tgt):
        plt.scatter(instance[0], instance[1], s=50, edgecolor='w',
                    alpha=0.5, color=colors[tgt])

    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1],
                    color='k', marker='x', lw=3, s=500)
    plt.title('Iris dataset partitioned by ABC - round 1')

    plt.subplot(1, 2, 2)
    centroids = dict(enumerate(decode_centroids(abc1.get_best_source(),
                                                n_clusters=3, data=data)))

    custom_tgt = []
    for instance in data:
        custom_tgt.append(assign_centroid(centroids, instance))

    colors = ['r', 'g', 'y']

    for instance, tgt in zip(data, custom_tgt):
        plt.scatter(instance[0], instance[1], s=50, edgecolor='w',
                    alpha=0.5, color=colors[tgt])

    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1],
                    color='k', marker='x', lw=3, s=500)
    plt.title('Iris dataset partitioned by ABC - final')

    plt.show()


def decode_centroids(centroids, n_clusters, data):
    return np.reshape(centroids, (n_clusters, data.shape[1]))


def assign_centroid(centroids, point):
    distances = [np.linalg.norm(point - centroids[idx]) for idx in centroids]
    return np.argmin(distances)


"""
    testing various colony sizes mixed with different ratios of employees to onlookers
    moreover, plotting the time taken to train & predict by the model in various colony sizes
"""


def plot_iris_testing_ratio_colony_sizes_time():
    datasets = {
        'iris': load_iris(),
    }
    colony_sizes = [10 * (i + 1) for i in range(13)]
    all_a = [[] for _ in range(7)]
    all_times = []
    for c_size in colony_sizes:
        for i in range(len(all_a)):
            dataset = datasets['iris']
            data = dataset.data
            X_train, X_test, y_train, y_test = train_test_split(data, dataset.target, test_size=0.25,
                                                                random_state=42, stratify=dataset.target)
            classifier = ABClassifier(colony_size=c_size, employee_to_onlooker_ratio=(i+1)*0.25)
            try:
                t0 = time.time()
                classifier.fit(X_train, y_train.tolist())

                labels = classifier.predict(X_test)
                t1 = time.time()
                print(confusion_matrix(y_true=y_test, y_pred=labels))
                all_a[i].append(accuracy_score(y_true=y_test, y_pred=labels))
                if i == 2:
                    all_times.append(t1-t0)
            except:
                all_a[i].append(0)
                if i == 2:
                    all_times.append(0)

    """
        plotting the graphs:
            the right is time taken to predict & train the model with default ratio (1.0)
            the left is the different ratios in various sizes of populations
          
    """
    plt.figure(1, figsize=(6, 3))

    plt.subplot(1, 2, 1)
    for i in range(len(all_a)):
        plt.plot(colony_sizes, all_a[i])
    plt.ylabel('accuracy')
    plt.xlabel('colony size')
    plt.legend(['1/2', '3/4', '1/1', '5/4', '3/2', '7/4', '2/1'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.bar(colony_sizes, all_times)
    plt.ylabel('time to train and predict')
    plt.xlabel('colony size')

    plt.show()


def clustering_illustration_abc():
    data = MinMaxScaler().fit_transform(load_iris()['data'][:, [1, 3]])
    objective_function1 = SSE(dim=6, n_centroids=3, data=data)
    abc1 = ABClustering(objective_function=objective_function1, colony_size=30, cycles=300, max_tries_employee=100,
                        max_tries_onlooker=100, processing_opt=None)

    abc1.optimize()

    d_s = {'cluster centers': [np.reshape(i, (3, 2)) for i, j in abc1.optimal_value_tracking],
           'SSE': [j for i, j in abc1.optimal_value_tracking]}
    pd.DataFrame.from_dict(d_s).to_csv('abc_illustration.csv')


if __name__ == '__main__':
    # clustering_with_abc()
    # plot_iris_testing_ratio_colony_sizes_time()
    clustering_illustration_abc()
