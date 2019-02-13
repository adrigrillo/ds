# -*- coding: utf-8 -*-
"""
majority_element.py
=================

Implementation of an algorithm to determine whether there is a majority element in array :math:`a`, i.e., an
element :math:`e` that occurs in array :math:`a` at least n=2 many times. If a majority element exists in array
:math:`a`, the algorithm also returns the element.
"""
from typing import List, Optional


def get_majority_item(array: List[int], mode: str = 'default') -> Optional[int]:
    """
    Implementation to determine whether there is a majority element in array.

    :param array: array to check
    :param mode: 'dc' uses the divide and conquer implementation. 'default' uses the counting implementation.
    Default: 'default'.
    :return: the majority element or None if it does not exist
    """
    if array is None or len(array) == 0:
        return None
    if mode is 'dc':
        return _get_majority_dc(array)
    elif mode is 'default':
        return _get_majority_def(array)
    else:
        raise ValueError('Execution mode not compatible')


def _get_majority_def(array: List[int]) -> Optional[int]:
    """
    Basic implementation that counts the times every item appears in the array and then check if the item
    that appears most do it with a frequency equal or higher than n/2

    :param array: array to check
    :return: the majority element or None if it does not exist
    """
    if len(array) == 0:
        return None
    counter = dict()
    for item in array:
        if item in counter:
            counter[item] += 1
        else:
            counter[item] = 1
    majority = max(counter, key=counter.get)
    if counter[majority] > len(array) // 2:
        return majority
    else:
        return None


def _get_majority_dc(array: List[int]) -> Optional[int]:
    """
    Divide and conquer implementation of the majority item algorithm. This

    :param array: array to
    :return: the majority element or None if it does not exist
    """
    array_length = len(array)
    if array_length == 1:
        return array[0]
    split_index = array_length // 2
    majority_left = _get_majority_dc(array[:split_index])
    majority_right = _get_majority_dc(array[split_index:])
    if majority_left == majority_right:
        return majority_left
    count_majority_left = 0
    count_majority_right = 0
    for item in array:
        if item == majority_left:
            count_majority_left += 1
        elif item == majority_right:
            count_majority_right +=1
    if count_majority_left > split_index:
        return majority_left
    elif count_majority_right > split_index:
        return majority_right
    else:
        return None
