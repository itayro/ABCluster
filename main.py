import matplotlib.pyplot as plt
from ClusteringAlgorithm.ABClustering import ABClustering
from ClusteringAlgorithm.ABClassifier import ABClassifier
from utils.ObjectiveFunction import *
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score


def run():
    objective_function = Rosenbrock(dim=30)
    abc = ABClustering(objective_function=objective_function, processing_opt='all_dimensions')

    plt.figure(figsize=(10, 7))

    abc.optimize()

    plt.plot([i for i in range(5000)], abc.get_optimization_path(), lw=0.5, label='Rosenbrock')
    plt.legend(loc='upper right')

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.xticks(rotation=45)
    plt.show()


def clustering_with_abc():
    data = MinMaxScaler().fit_transform(load_iris()['data'][:, [1, 3]])
    objective_function1 = SSE(dim=6, n_centroids=3, data=data)
    abc1 = ABClustering(objective_function=objective_function1, colony_size=30, cycles=300, max_trials=100,
                        processing_opt='all_dimensions')

    abc1.optimize()

    centroids = dict(enumerate(decode_centroids(abc1.get_best_source(),
                                                n_clusters=3, data=data)))

    custom_tgt = []
    for instance in data:
        custom_tgt.append(assign_centroid(centroids, instance))

    colors = ['r', 'g', 'y']
    plt.figure(figsize=(9, 8))
    for instance, tgt in zip(data, custom_tgt):
        plt.scatter(instance[0], instance[1], s=50, edgecolor='w',
                    alpha=0.5, color=colors[tgt])

    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1],
                    color='k', marker='x', lw=5, s=500)
    plt.title('Partitioned Data found by ABC')
    plt.show()


def decode_centroids(centroids, n_clusters, data):
    return np.reshape(centroids, (n_clusters, data.shape[1]))


def assign_centroid(centroids, point):
    distances = [np.linalg.norm(point - centroids[idx]) for idx in centroids]
    return np.argmin(distances)



def try_classify():
    datasets = {
        'iris': load_iris(),
        'digits': load_digits(),
        'wine': load_wine(),
        'breast_cancer': load_breast_cancer(),

    }
    dataset = datasets['iris']
    data = MinMaxScaler().fit_transform(dataset.data)
    X_train, X_test, y_train, y_test = train_test_split(data, dataset.target, test_size=0.33,
                                                        random_state=42, stratify=dataset.target)

    classifier = ABClassifier()
    classifier.fit(X_train, y_train.tolist())

    labels = classifier.predict(X_test)

    print(confusion_matrix(y_true=y_test, y_pred=labels))
    print(accuracy_score(y_true=y_test, y_pred=labels))


if __name__ == '__main__':
    # run()
    # clustering_with_abc()
    try_classify()
