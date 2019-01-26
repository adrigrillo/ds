# -*- coding: utf-8 -*-
"""
k_means.py
=================

Implementation of the k-means algorithm
"""
import matplotlib.pyplot as plt
import numpy as np


class KMeans(object):
    def __init__(self, k: int = 10, tolerance: int = 1e-4):
        self.k = k
        self.centroids = None
        self._tolerance = tolerance

    def initialize_centroids(self, points: np.ndarray) -> np.ndarray:
        """
        Centroid initialization. This method select k random points of the data using sampling without replacement
        and set them as the initial centroids.

        :param points: data points to cluster
        """
        self.centroids = points[np.random.choice(points.shape[0], self.k, replace=False)]
        return self.centroids

    def assign_centroid(self, points: np.ndarray) -> np.ndarray:
        """
        Methods that returns the closest centroid of the points. For each point the distance with respect the
        centroids are calculated using the manhattan distance, assigning each point to the centroid with is
        nearer.

        :param points: data points to cluster
        :return: list with the nearest centroid of the points
        """
        distance = np.sqrt(np.sum(np.power(points - np.expand_dims(self.centroids, axis=1), 2), axis=2))
        return np.argmin(distance, axis=0)

    def move_centroids(self, points: np.ndarray, closest_centroids: np.ndarray) -> np.ndarray:
        """
        Method that calculate centroids of the points assigned to each cluster.

        :param points: data points to cluster
        :param closest_centroids: list with the nearest centroid of the points
        :return: new centroids
        """
        for i in range(self.centroids.shape[0]):
            self.centroids[i] = np.mean(points[closest_centroids == i], axis=0)
        return self.centroids

    def fit(self, points: np.ndarray, iterations: int = 300) -> None:
        """
        Compute k-means clustering for the given iterations or until there is no improvement
        :param points: data points to cluster
        :param iterations: number of times the centroids will be calculated. Default: 300.
        """
        if self.centroids is None:
            last_centroids = self.initialize_centroids(points)
        else:
            last_centroids = self.centroids
        for i in range(iterations):
            closest_centroids = self.assign_centroid(points)
            new_centroids = self.move_centroids(points, closest_centroids)
            if np.sum(np.power(new_centroids - last_centroids, 2), axis=(1, 0)) > self._tolerance:
                last_centroids = new_centroids
            else:
                break

    def plot_centroids(self, points: np.ndarray) -> None:
        plt.subplot(122)
        plt.scatter(points[:, 0], points[:, 1])
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='r', s=100)
        plt.show()


if __name__ == '__main__':
    # points = np.array(['1','2','3'])
    points = np.vstack(((np.random.randn(150, 2) * 0.75 + np.array([1, 0])),
                        (np.random.randn(50, 2) * 0.25 + np.array([-0.5, 0.5])),
                        (np.random.randn(50, 2) * 0.5 + np.array([-0.5, -0.5]))))
    kmeans = KMeans(k=3)
    centroids = kmeans.initialize_centroids(points)
    plt.subplot(121)
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)

    closest = kmeans.assign_centroid(points)
    centroids = kmeans.move_centroids(points, closest)
    plt.subplot(122)
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
    plt.show()

    kmeans.fit(points)
    plt.scatter(points[:, 0], points[:, 1])
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', s=100)
    plt.show()
