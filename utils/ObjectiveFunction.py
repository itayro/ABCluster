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
