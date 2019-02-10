from src.algorithms.majority_element import get_majority_item


def test_no_array():
    array = None
    assert get_majority_item(array) is None


def test_empty_array():
    array = []
    assert get_majority_item(array) is None


################
# DEFAULT MODE #
################


def test_one_element_array():
    array = [1]
    majority_item = get_majority_item(array)
    assert majority_item is 1


def test_majority_array():
    array = [1, 2, 1, 2, 1, 2, 2]
    majority_item = get_majority_item(array)
    assert majority_item is 2
    assert majority_item is not 1


def test_no_majority_array():
    array = [1, 2, 1, 2, 1, 2, 3]
    majority_item = get_majority_item(array)
    assert majority_item is None


######################
# DIVIDE AND CONQUER #
######################

def test_dc_one_element_array():
    array = [1]
    majority_item = get_majority_item(array, 'dc')
    assert majority_item is 1


def test_dc_majority_array():
    array = [1, 2, 1, 2, 1, 2, 2]
    majority_item = get_majority_item(array, 'dc')
    assert majority_item is 2
    assert majority_item is not 1


def test_dc_no_majority_array():
    array = [1, 2, 1, 2, 1, 2, 3]
    majority_item = get_majority_item(array, 'dc')
    assert majority_item is None
