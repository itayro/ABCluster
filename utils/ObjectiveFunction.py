from abc import ABCMeta
from abc import abstractmethod
from six import add_metaclass
import numpy as np
from scipy import optimize


@add_metaclass(ABCMeta)
class ObjectiveFunction(object):
    """
        Objective Function is the function we intend to optimize using the ABC algorithm

        min_lim - the minimum limit of definition of the objective function (in each dimension)
        max_lim - the maximum limit of definition of the objective function (in each dimension)
        dim - the number of dimensions of the instance we are evaluating

        other words:
            f(x1,x2,...xn) -> dim == n && foreach 1<=i<=n: min_lim <= xi <= max_lim
    """
    def __init__(self, min_lim, max_lim, dim):
        self.min_lim = min_lim
        self.max_lim = max_lim
        self.dim = dim

    def random_sample(self):
        return list(np.random.uniform(self.min_lim, self.max_lim, self.dim))

    @abstractmethod
    def evaluate(self, instance):
        pass

    def get_min_lim(self):
        return self.min_lim

    def get_max_lim(self):
        return self.max_lim

    def get_dim(self):
        return self.dim


class Sphere(ObjectiveFunction):
    """
        f(x1,..xD) = x1^2 + x2^2 + ... + xD^2

        the limit used is [-100,100]
    """
    def __init__(self, dim):
        ObjectiveFunction.__init__(self, -100.0, 100.0, dim)

    def evaluate(self, instance):
        return sum(np.power(instance, 2))


class Rosenbrock(ObjectiveFunction):
    """
        f(x1,x2,...xn) = {foreach 1<=i<=n-1 100(x_i+1 - x_i)^2 + (x_i - 1)^2}
    """
    def __init__(self, dim):
        ObjectiveFunction.__init__(self, -30.0, 30.0, dim)

    def evaluate(self, instance):
        return optimize.rosen(instance)


class SSE(ObjectiveFunction):

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
        centroids = np.reshape(instance, (self.n_centroids, self.dim))

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
        return sum([np.power(distances, 2) for center_id, distances in self.distances.items()]) / len(self.data)

    def evaluate(self, instance):
        self.clusters = dict(enumerate([[] for i in range(self.n_centroids)]))
        self.distances = dict(enumerate([[] for i in range(self.n_centroids)]))

        self.__set_centroids(instance)

        self.__assign_data_to_clusters()

        return self.__calc_sse()
