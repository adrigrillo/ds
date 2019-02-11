import pytest

from src.algorithms.interval_coverage import check_interval_coverage, get_minimum_subintervals_set_math


#############################
# TEST FOR SOLUTION CHECKER #
#############################

def test_wrong_upper_bound():
    with pytest.raises(ValueError):
        check_interval_coverage(0, [1, 3, 5])


def test_no_overlapping_points():
    assert check_interval_coverage(5, [1, 3, 5]) is True


def test_overlapping_points():
    assert check_interval_coverage(5, [1, 2, 3, 4, 5]) is True


def test_wrong_solution():
    assert check_interval_coverage(5, [2, 4, 5]) is False


def test_no_overlapping_points_range_2():
    assert check_interval_coverage(10, [3, 6, 9], range_length=3) is True


def test_overlapping_points_range_2():
    assert check_interval_coverage(10, [2, 5, 7], range_length=3) is True


def test_wrong_solution_range_2():
    assert check_interval_coverage(10, [4, 8], range_length=3) is False


#################################
# TEST FOR SOLUTION FINDER MATH #
#################################

def test_wrong_upper_bound_math():
    with pytest.raises(ValueError):
        get_minimum_subintervals_set_math(0)


def test_no_overlapping_case_math():
    result = get_minimum_subintervals_set_math(4, range_length=1)
    assert result == [1, 3]


def test_out_of_bound_case_math():
    result = get_minimum_subintervals_set_math(5, range_length=2)
    assert result == [2, 5]
