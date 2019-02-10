# -*- coding: utf-8 -*-
"""
hierarchical_clustering.py
=================

This module contains an implementation to calculate the hierarchical clustering of set of points using the
distance matrix.

The distance matrix is a square matrix (two-dimensional array) containing the distances, taken
pairwise, between the elements of a set. An example of a distance matrix would be:
np.array([[0.0, 0.3, 0.4, 0.7],
          [0.3, 0.0, 0.5, 0.8],
          [0.4, 0.5, 0.0, 0.45],
          [0.7, 0.8, 0.45, 0.0]])
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn import manifold


def approx_points_from_distance_matrix(distance_matrix: np.ndarray, dimensions: int) -> np.ndarray:
    """
    Returns some coordinates that approximate the distance matrix.

    :param distance_matrix: precomputed distance matrix
    :param dimensions: number of dimensions in the points
    :return: array with the points that fulfill the distance matrix constrains
    """
    mds = manifold.MDS(n_components=dimensions, dissimilarity='precomputed')
    results = mds.fit(distance_matrix)
    return results.embedding_


def hierarchical_clustering_from_distance(distance_matrix: np.ndarray, dimensions: int = 2,
                                          method: str = 'single', plot: bool = True) -> np.ndarray:
    """
    Method that performs hierarchical clustering from a distance matrix. This method generates the
    points corresponding to the distance matrix introduced as input and performs the hierarchical
    clustering over this points.

    The hierarchical algorithm used is from the sklearn package
    The following are methods for calculating the distance between the
    newly formed cluster :math:`u` and each :math:`v`.

      * 'single' assigns

        .. math:: d(u,v) = \\min(dist(u[i],v[j]))

        for all points :math:`i` in cluster :math:`u` and
        :math:`j` in cluster :math:`v`. This is also known as the
        Nearest Point Algorithm.

      * 'complete' assigns

        .. math:: d(u, v) = \\max(dist(u[i],v[j]))

        for all points :math:`i` in cluster u and :math:`j` in
        cluster :math:`v`. This is also known by the Farthest Point
        Algorithm or Voor Hees Algorithm.

      * 'average' assigns

        .. math:: d(u,v) = \\sum_{ij} \\frac{d(u[i], v[j])}{(|u|*|v|)}

        for all points :math:`i` and :math:`j` where :math:`|u|`
        and :math:`|v|` are the cardinalities of clusters :math:`u`
        and :math:`v`, respectively. This is also called the UPGMA
        algorithm.

      * 'weighted' assigns

        .. math:: d(u,v) = (dist(s,v) + dist(t,v))/2

        where cluster u was formed with cluster s and t and v
        is a remaining cluster in the forest. (also called WPGMA)

      * 'centroid' assigns

        .. math:: dist(s,t) = ||c_s-c_t||_2

        where :math:`c_s` and :math:`c_t` are the centroids of
        clusters :math:`s` and :math:`t`, respectively. When two
        clusters :math:`s` and :math:`t` are combined into a new
        cluster :math:`u`, the new centroid is computed over all the
        original objects in clusters :math:`s` and :math:`t`. The
        distance then becomes the Euclidean distance between the
        centroid of :math:`u` and the centroid of a remaining cluster
        :math:`v` in the forest. This is also known as the UPGMC
        algorithm.

      * 'median' assigns :math:`d(s,t)` like the ``centroid``
        method. When two clusters :math:`s` and :math:`t` are combined
        into a new cluster :math:`u`, the average of centroids s and t
        give the new centroid :math:`u`. This is also known as the
        WPGMC algorithm.

      * 'ward' uses the Ward variance minimization algorithm. This is also
        known as the incremental algorithm.

    For more information visit:
    https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering

    :param distance_matrix: matrix with the full distances matrix, containing zeros in the diagonal.
    :param dimensions: dimensions in the data.
    :param method: way of calculating the inter-cluster distance.
    :param plot: flag to plot the dendrogram.
    :return: the hierarchical clustering encoded as a linkage matrix.
    """
    points = approx_points_from_distance_matrix(distance_matrix, dimensions=dimensions)
    links = linkage(points, method=method)
    if plot:
        labels = np.array(list('x_{0}'.format(i + 1) for i in range(len(points))))
        plt.figure()
        dendrogram(links, labels=labels)
        plt.show()
    return links


if __name__ == '__main__':
    dist_matrix = np.array([[0.0, 0.3, 0.4, 0.7],
                            [0.3, 0.0, 0.5, 0.8],
                            [0.4, 0.5, 0.0, 0.45],
                            [0.7, 0.8, 0.45, 0.0]])
    hierarchical_clustering_from_distance(dist_matrix, method='complete')
