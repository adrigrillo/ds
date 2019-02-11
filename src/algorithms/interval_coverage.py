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


def check_interval_coverage(intvl_upper_limit: int, sub_intvl_positions: List[int], range: int = 1) -> bool:
    if intvl_upper_limit == 0:
        raise ValueError('The upper limit of the interval has to be bigger than 0')
    covered = 0
    for position in sub_intvl_positions:
        lower_bound = position - range
        upper_bound = position + range
        if lower_bound <= covered < upper_bound:
            covered = upper_bound
    if covered > intvl_upper_limit:
        return True
    else:
        return False


# def minimum_number_of_stations(intvl_upper_limit: int, range: int = 1) -> float:
#     return math.log(2 * intvl_upper_limit / range - 4, 2)


if __name__ == '__main__':
    result = check_interval_coverage(5, [2, 4, 5])
    print(result)
