import networkx as nx

from src.algorithms.weighted_max_indep_set import maximum_independent_set


def test_simple_case():
    g = nx.Graph()
    g.add_nodes_from([0, 3, 4], weight=3)
    g.add_nodes_from([2, 5], weight=1)
    g.add_node(1, weight=5)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert maximum_independent_set(g) == {1, 3, 5}


def test_simple_case_2():
    g = nx.Graph()
    g.add_nodes_from([0, 3, 4], weight=3)
    g.add_nodes_from([2, 5], weight=1)
    g.add_node(1, weight=2)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert maximum_independent_set(g) == {0, 2, 4}


def test_jump_case():
    g = nx.Graph()
    g.add_nodes_from([0, 2, 3, 5], weight=1)
    g.add_nodes_from([1, 4], weight=5)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    assert maximum_independent_set(g) == {1, 4}


def test_jump_case_relaxed():
    g = nx.Graph()
    g.add_nodes_from([0, 2, 3, 4, 6], weight=1)
    g.add_nodes_from([1, 5], weight=5)
    g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
    assert maximum_independent_set(g) == {1, 3, 5}
