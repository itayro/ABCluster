import matplotlib.pyplot as plt
from ClusteringAlgorithm.ABClustering import ABClustering
from utils.ObjectiveFunction import *

from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler


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


if __name__ == '__main__':
    # run()
    clustering_with_abc()
