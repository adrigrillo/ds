from src.algorithms.maximum_subarray import find_optimum_subarray


def test_no_array():
    array = None
    assert find_optimum_subarray(array) == []


def test_empty_array():
    array = []
    assert find_optimum_subarray(array) == []


def test_one_positive_element_array():
    array = [1]
    optimum_subarray = find_optimum_subarray(array)
    assert optimum_subarray == [1]


def test_one_negative_element_array():
    array = [-10]
    optimum_subarray = find_optimum_subarray(array)
    assert optimum_subarray == []


def test_default_case():
    array = [1, 2, -5, 4, 8, 6, -19, 1, 2, 3, 5, 4, 4, 2]
    optimum_subarray = find_optimum_subarray(array)
    assert optimum_subarray == [1, 2, 3, 5, 4, 4, 2]


def test_all_array():
    array = [1, 2, 1, 2, 1, 2, 3]
    optimum_subarray = find_optimum_subarray(array)
    assert optimum_subarray == array


def test_no_optimum_array():
    array = [1, -2, 1, -2, 1, -2, 3]
    optimum_subarray = find_optimum_subarray(array)
    assert optimum_subarray == [3]
