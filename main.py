import matplotlib.pyplot as plt
from ClusteringAlgorithm.ABClustering import ABClustering
from utils.ObjectiveFunction import *


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


if __name__ == '__main__':
    run()
