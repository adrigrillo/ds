# -*- coding: utf-8 -*-
"""
interval_coverage.py
=================

Implementation of an algorithm to check if the given solution for the coverage of an interval is correct.
In this problem, an interval :math:`[0,N]` will be provide with a list of the middle points of the sub-intervals.
Each sub-interval will cover a range :math:`R` so a position will cover :math:`[pos-R,pos+R]`.
The algorithm will return if the position of the sub-interval is correct to cover the full original interval.
"""
import math
from typing import List


def check_interval_coverage(intvl_upper_limit: int, sub_intvl_positions: List[int], range_length: int = 1) -> bool:
    """
    Method that checks if given sub-intervals are correctly situated to cover all the original segment.

    :param intvl_upper_limit: upper bound of the interval
    :param sub_intvl_positions: middle point positions of the sub-intervals
    :param range_length: range of the sub-interval from the middle point
    :return: True is the solution is valid, false otherwise
    """
    if intvl_upper_limit == 0:
        raise ValueError('The upper limit of the interval has to be bigger than 0')
    covered = 0
    for position in sub_intvl_positions:
        lower_bound = position - range_length
        upper_bound = position + range_length
        if lower_bound <= covered < upper_bound:
            covered = upper_bound
    if covered >= intvl_upper_limit:
        return True
    else:
        return False


def get_minimum_subintervals_set_math(intvl_upper_limit: int, range_length: int = 1) -> List[int]:
    """
    Method that returns the minimal set of placements that cover the full interval. This method uses
    the mathematical expresion that minimizes the number of points, however it can go out of the range,
    so the last point is set to the upper bound when this situation happens.

    :param intvl_upper_limit: upper bound of the interval
    :param range_length: range of the sub-interval from the placement points
    :return: list with the position
    """
    if intvl_upper_limit == 0:
        raise ValueError('The upper limit of the interval has to be bigger than 0')
    number_of_subintervals = math.ceil(intvl_upper_limit / (2 * range_length))
    positions = []
    for i in range(number_of_subintervals):
        if i == 0:
            positions.append(range_length)
        else:
            position = range_length + 2 ** i * range_length
            positions.append(position if position <= intvl_upper_limit else intvl_upper_limit)
    return positions


if __name__ == '__main__':
    result = get_minimum_subintervals_set_math(5, 2)
    print(result)
