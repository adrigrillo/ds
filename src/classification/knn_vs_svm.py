# -*- coding: utf-8 -*-
"""
knn_vs_svm.py
=================

Comparison between k-NN and SVM classification algorithms with leave-one-out cross validation
"""
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm, neighbors
from sklearn.model_selection import LeaveOneOut


def make_meshgrid(x: np.ndarray, y: np.ndarray, h: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a mesh of points to plot in

    :param x: data to base x-axis meshgrid on
    :param y: data to base y-axis meshgrid on
    :param h: stepsize for meshgrid, optional
    :return: xx and yy
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx: np.ndarray, yy: np.ndarray, **params):
    """
    Generates a plot with the decision boundaries for a classifier.

    :param ax: matplotlib axes object
    :param clf: a classifier
    :param xx: meshgrid ndarray
    :param yy: meshgrid ndarray
    :param params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_decision_boundaries(x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                             classifiers: Tuple) -> None:
    """
    Method that generates a plot with the decision boundaries of both of the classifiers, comparing one with the
    other.

    :param x: data points
    :param y: class of the data points
    :param classifiers: tuple with the classifiers being compared
    """
    titles = ('K-NN', 'SVM')
    # Set-up 2x1 grid for plotting.
    fig, sub = plt.subplots(1, 2)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    # Generate the grid
    x0, x1 = x[:, 0], x[:, 1]
    xx, yy = make_meshgrid(x0, x1)
    # Plot the different models
    for clf, title, ax in zip(classifiers, titles, sub.flatten()):
        plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        ax.scatter(x0, x1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k', label='Training instance')
        ax.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='w',
                   label='Test instance')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Axis X')
        ax.set_ylabel('Axis Y')
        ax.set_xticks(())
        ax.set_yticks(())
        ax.set_title(title)
    plt.legend(loc='best')
    plt.show()


def train_and_count_errors(x: np.ndarray, y: np.ndarray, neighbours: int = 1, plot=True) -> Tuple[float, float]:
    """
    Method that trains the classifiers using the leave-one-out method and counts the errors committed classifying
    the test instance.

    :param x: data points
    :param y: class of the data points
    :param neighbours: neighbours used in the k-NN algorithm
    :param plot: flag to plot the decision boundaries of the classificators
    :return: tuple with the errors accumulated by the classificators. Tuple[errors_knn, errors_svm]
    """
    leave_one_out = LeaveOneOut()
    accumulated_knn_error = 0
    accumulated_svm_error = 0
    for train_index, test_index in leave_one_out.split(x):
        knn_clf = neighbors.KNeighborsClassifier(neighbours, metric='euclidean')
        svm_clf = svm.SVC(kernel='linear', C=10)
        # Train the classifiers
        knn_clf.fit(x[train_index], y[train_index])
        svm_clf.fit(x[train_index], y[train_index])
        # Predict the test instance
        if knn_clf.predict(x[test_index]) != y[test_index]:
            accumulated_knn_error += 1
        if svm_clf.predict(x[test_index]) != y[test_index]:
            accumulated_svm_error += 1
        if plot:
            classifiers = (knn_clf, svm_clf)
            plot_decision_boundaries(x[train_index], y[train_index], x[test_index], y[test_index], classifiers)
    return accumulated_knn_error, accumulated_svm_error


if __name__ == '__main__':
    data = np.array([[1, 1], [2, 1], [10, 1], [11, 1], [1, 10], [2, 10], [10, 10], [11, 10]])
    target_class = np.array([0, 0, 1, 1, 1, 1, 0, 0])
    knn_error, svm_error = train_and_count_errors(data, target_class)
    print('The k-NN classifier has made {0} error, the SVM has made {1} errors'.format(knn_error, svm_error))
    data = np.array([[1, 1], [2, 1], [10, 1], [11, 1], [1, 10], [2, 10], [10, 10], [11, 10]])
    target_class = np.array([0, 0, 0, 1, 0, 0, 0, 1])
    knn_error, svm_error = train_and_count_errors(data, target_class)
    print('The k-NN classifier has made {0} error, the SVM has made {1} errors'.format(knn_error, svm_error))
