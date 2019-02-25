import matplotlib.pyplot as plt
from ClusteringAlgorithm.ABClustering import ABClustering
from ClusteringAlgorithm.ABClassifier import ABClassifier
from utils.ObjectiveFunction import *
from utils.ClusteringObjectiveFunction import *
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, load_wine
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import time


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
    abc1 = ABClustering(objective_function=objective_function1, colony_size=30, cycles=300, max_tries_employee=100,
                        max_tries_onlooker=100, processing_opt=None)

    abc1.optimize()

    # plt.plot([i for i in range(300)], abc1.get_optimization_path())
    # plt.title('SSE over cycles')
    # plt.xlabel('cycle number')
    # plt.ylabel('SSE')
    # plt.show()

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


def try_classify():
    datasets = {
        'iris': load_iris(),
        'digits': load_digits(),
        'wine': load_wine(),
        'breast_cancer': load_breast_cancer(),

    }
    dataset = datasets['iris']
    # data = MinMaxScaler().fit_transform(dataset.data)
    data = dataset.data
    X_train, X_test, y_train, y_test = train_test_split(data, dataset.target, test_size=0.33,
                                                        random_state=42, stratify=dataset.target)

    classifier = ABClassifier()
    classifier.fit(X_train, y_train.tolist())

    labels = classifier.predict(X_test)

    print(confusion_matrix(y_true=y_test, y_pred=labels))
    print(accuracy_score(y_true=y_test, y_pred=labels))


def plot_params():
    datasets = {
        'iris': load_iris(),
    }
    colony_sizes = [10*(i+1) for i in range(10)]
    accuracies = []
    for c_size in colony_sizes:
        dataset = datasets['iris']
        data = dataset.data
        X_train, X_test, y_train, y_test = train_test_split(data, dataset.target, test_size=0.25,
                                                            random_state=42, stratify=dataset.target)
        classifier = ABClassifier(colony_size=c_size)
        classifier.fit(X_train, y_train.tolist())

        labels = classifier.predict(X_test)

        print(confusion_matrix(y_true=y_test, y_pred=labels))
        accuracies.append(accuracy_score(y_true=y_test, y_pred=labels))

    plt.plot(colony_sizes, accuracies)
    plt.ylabel('accuracy')
    plt.xlabel('colony size')
    plt.show()


def plot_2():
    datasets = {
        'iris': load_iris(),
    }
    colony_sizes = [10 * (i + 1) for i in range(13)]
    accuracies = []
    accuracies_half = []
    accuracies_double = []
    accuracies_three_quarters = []
    accuracies_five_quarters = []
    accuracies_six_quarters = []
    accuracies_seven_quarters = []
    all_a = [accuracies_half, accuracies_three_quarters, accuracies, accuracies_five_quarters,
             accuracies_six_quarters, accuracies_seven_quarters, accuracies_double]
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

    plt.figure(1, figsize=(6, 3))

    plt.subplot(1, 2, 1)
    plt.plot(colony_sizes, all_a[0])
    plt.plot(colony_sizes, all_a[1])
    plt.plot(colony_sizes, all_a[2])
    plt.plot(colony_sizes, all_a[3])
    plt.plot(colony_sizes, all_a[4])
    plt.plot(colony_sizes, all_a[5])
    plt.plot(colony_sizes, all_a[6])
    plt.ylabel('accuracy')
    plt.xlabel('colony size')
    plt.legend(['1/2', '3/4', '1/1', '5/4', '3/2', '7/4', '2/1'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.bar(colony_sizes, all_times)
    plt.ylabel('time to train and predict')
    plt.xlabel('colony size')

    plt.show()


def plot_3():
    dataset = load_iris()
    objective_function = SSE()
    abc = ABClustering(objective_function=objective_function, processing_opt='all_dimensions')

    plt.figure(figsize=(10, 7))

    abc.optimize()

    plt.plot([i for i in range(5000)], abc.get_optimization_path(), lw=0.5, label='SSE')
    plt.legend(loc='upper right')

    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    # run()
    clustering_with_abc()
    # try_classify()
    # plot_params()
    # plot_2()
