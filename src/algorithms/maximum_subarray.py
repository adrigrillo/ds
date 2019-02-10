# -*- coding: utf-8 -*-
"""
maximum_subarray.py
=================

Implementation of an algorithm to solve the maximum subarray problem. It is that given an array :math:`a` of :math:`n`
integers, asks to find a non-empty contiguous subarray :math:`a[s, ..., e]` that has the largest sum of its
elements.
"""
from typing import List, Optional


def find_optimum_subarray(array: List[int]) -> List[int]:
    """
    Method that search for the subarray with contiguous elements that maximizes the sum of its elements.

    :param array: array to search
    :return: optimum subarray or empty if it does not exist
    """
    if array is None or len(array) == 0:
        return []
    elif len(array) == 1 and array[0] > 0:
        # only one element and positive
        return array
    i_init: int = 0
    i_end: int = 0
    best_i_init: int = 0
    best_i_end: int = 0
    subarray_sum: int = 0
    best_sum: int = 0
    last_index = len(array) - 1
    while i_init <= last_index and i_end <= last_index:
        if subarray_sum + array[i_end] > 0:
            subarray_sum += array[i_end]
            i_end += 1
        else:
            subarray_sum = 0
            i_end += 1
            i_init = i_end
        if subarray_sum > best_sum:
            best_sum = subarray_sum
            best_i_init = i_init
            best_i_end = i_end
    return array[best_i_init:best_i_end]
