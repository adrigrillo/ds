from typing import Set, List

import networkx as nx


def maximum_independent_set(graph: nx.Graph) -> Set[int]:
    """
    Method that calculates the maximum independent set that maximizes the weighted sum of a path graph.
    :param graph: path graph with weigths on the vertices
    :return: indices of the vertices that form the maximum independent set
    """
    if len(graph) == 0:
        return set()
    weights = _get_set_values(graph)
    return _reconstruct_set_from_weights(weights)


def _reconstruct_set_from_weights(weights: List[int]) -> Set[int]:
    """
    Method that uses the auxiliary array of weights to reconstruct the maximum independent set. It compare the
    weighted matrix iteratively starting for the last vertex of the graph.
    The algorithm is the next:

    * If the weighted sum of the set with vertex :math:`i` is different than the weighted sum of the set
      without the vertex, the vertex is added to the set and the previous one is skipped.
    * Else, the vertex is not added and the previous vertex is analyze

    :param weights: auxiliary array with the weighted sum
    :return: a set with the indices of the vertices that form the maximum independent set
    """
    vertex_index = len(weights) - 1
    max_independent_set = set()
    while vertex_index >= 1:
        if weights[vertex_index] != weights[vertex_index - 1]:
            max_independent_set.add(vertex_index)
            vertex_index -= 2
        else:
            vertex_index -= 1
    # If the second vertex is not in the set, add the first
    if not max_independent_set.__contains__(1):
        max_independent_set.add(0)
    return max_independent_set


def _get_set_values(graph: nx.Graph) -> List[int]:
    """
    Iterative method that saves in an auxiliary array the weighted sum of the set. This method uses the
    first vertex and in every step add one vertex to the set. If the new vertex improves the weighted sum
    of the last set, the new weighted sum is saved in the auxiliary array with the index of the vertex.
    If the weighted sum is equal or worse that previous one, the previous is saved.

    :param graph: graph to analyze, with weighted vertices
    :return: weighted sum of the different sets
    """
    weights = list()
    for vertex_index in range(len(graph)):
        if vertex_index - 2 >= 0:
            weight_new_set = weights[vertex_index - 2] + graph.nodes[vertex_index]['weight']
            weights.append(_compare_weights_of_sets(weight_new_set, weights[vertex_index - 1]))
        elif vertex_index - 1 == 0:
            weight_new_vertex = graph.nodes[vertex_index]['weight']
            weights.append(_compare_weights_of_sets(weight_new_vertex, weights[vertex_index - 1]))
        else:
            weights.append(graph.nodes[vertex_index]['weight'])
    return weights


def _compare_weights_of_sets(weight_new_set: int, weight_old_set: int) -> int:
    """
    Method that compares the weighted sum of two sets and keeps the bigger one.

    :param weight_new_set: weighted sum of the new set
    :param weight_old_set: weighted sum of the old set
    :return: bigger weighted sum
    """
    if weight_new_set > weight_old_set:
        return weight_new_set
    else:
        return weight_old_set
