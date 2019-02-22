from ClusteringAlgorithm.ABClustering import ABClustering
from utils.ObjectiveFunction import SSE
import numpy as np


class ABClassifier:
    def __init__(self, colony_size=20, cycles=5000, max_trials=100,
                 processing_opt=None, employee_to_onlooker_ratio=1.0):

        self._cluster_algorithm = ABClustering(objective_function=None, colony_size=colony_size,
                                               cycles=cycles, max_trials=max_trials,
                                               processing_opt=processing_opt,
                                               employee_to_onlooker_ratio=employee_to_onlooker_ratio)
        self._centroids_to_labels = {}
        self._objective_function = None

    """
        requirements:
            train_data - numpy.ndarray
            labels - non-negative list of numbers
    """
    def fit(self, train_data, labels):
        print('start fit')
        num_of_labels = len(set(labels))
        self._objective_function = SSE(train_data.shape[1] * num_of_labels,
                                       train_data.tolist(),
                                       num_of_labels)
        self._cluster_algorithm.set_objective_function(self._objective_function)
        self._cluster_algorithm.optimize()

        self._centroids_to_labels = self._objective_function.map_centroid_to_labels(train_data.tolist(), labels)
        print('end fit')

    def predict(self, data_points):
        print('start predict')
        centroids = self._objective_function.get_centroids()
        labels = []
        for data_point in data_points:

            min_centroid_id = None
            min_norm = np.inf

            for centroid_id in centroids.keys():
                norm = np.linalg.norm(data_point - centroids[centroid_id])

                if min_centroid_id is None or norm < min_norm:
                    min_norm = norm
                    min_centroid_id = centroid_id

            labels.append(self._centroids_to_labels[min_centroid_id])
        print('end predict')
        return labels
