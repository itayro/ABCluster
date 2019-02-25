from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass
import numpy as np
from utils.ObjectiveFunction import ObjectiveFunction


@add_metaclass(ABCMeta)
class ClusteringObjectiveFunction(ObjectiveFunction):
    """
       interface for objective functions related to clustering
    """
    def __init__(self, min_lim, max_lim, dim):
        ObjectiveFunction.__init__(self, min_lim, max_lim, dim)

    @abstractmethod
    def evaluate(self, instance):
        pass

    @abstractmethod
    def map_centroid_to_labels(self, data, labels):
        pass

    @abstractmethod
    def get_centroids(self):
        pass


class SSE(ClusteringObjectiveFunction):

    """
        the objective function used for clustering (sum of square errors)
        in words:
            sum of the expression bellow (all clusters and all the points in them)

            for each cluster:
                for each point in the cluster k:
                    ||cluster_loc - point||^2
    """

    def __init__(self, dim, data, n_centroids):
        ObjectiveFunction.__init__(self, 0.0, 1.0, dim)
        self.centroids = {}
        self.data = data
        self.n_centroids = n_centroids
        self.clusters = None
        self.distances = None

    def __set_centroids(self, instance):
        centroids = np.reshape(instance, (self.n_centroids, self.dim // self.n_centroids))

        for ind, centroid in enumerate(centroids):
            self.centroids[ind] = centroid

    def __assign_data_to_clusters(self):
        for data_point in self.data:
            min_centroid_id = None
            min_norm = np.inf

            for centroid_id in self.centroids.keys():
                norm = np.linalg.norm(data_point - self.centroids[centroid_id])

                if min_centroid_id is None or norm < min_norm:
                    min_norm = norm
                    min_centroid_id = centroid_id

            self.clusters[min_centroid_id].append(data_point)
            self.distances[min_centroid_id].append(min_norm)

    def __calc_sse(self):
        return sum([sum(np.power(distances, 2)) for center_id, distances in self.distances.items()]) / len(self.data)

    def evaluate(self, instance):
        self.clusters = dict(enumerate([[] for i in range(self.n_centroids)]))
        self.distances = dict(enumerate([[] for i in range(self.n_centroids)]))

        self.__set_centroids(instance)

        self.__assign_data_to_clusters()

        return self.__calc_sse()

    def map_centroid_to_labels(self, data, labels):
        centroid_label = dict(enumerate([[] for i in range(self.n_centroids)]))

        for data_point, label in zip(data, labels):
            min_centroid_id = None
            min_norm = np.inf

            for centroid_id in self.centroids.keys():
                norm = np.linalg.norm(data_point - self.centroids[centroid_id])

                if min_centroid_id is None or norm < min_norm:
                    min_norm = norm
                    min_centroid_id = centroid_id

            centroid_label[min_centroid_id].append(label)

        centroid_to_label = {}
        for centroid_id, centroid_labels in centroid_label.items():
            centroid_to_label[centroid_id] = np.argmax(np.bincount(np.array(centroid_labels)))

        return centroid_to_label

    def get_centroids(self):
        return self.centroids
