from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def sampling_with_replacement(num_diff_items: int = 10, num_samples: int = 100) -> np.ndarray:
    """
    This method samples with replacement a given number of times for a bag containing the indicated number
    of different items.

    :param num_diff_items: number of different items that the bag will contain
    :param num_samples: number of samples to take from the bag
    :return: array with the samples
    """
    bag: np.ndarray = np.arange(num_diff_items)
    picks: np.ndarray = np.zeros(num_samples)
    for i in range(num_samples):
        picks[i] = np.random.choice(bag)
    return picks


def execute_series_of_sampling(num_diff_items: int = 10, num_samples: int = 100,
                               num_of_series: int = 100) -> np.ndarray:
    """
    Method that executes a given number of times the sampling with replacement and returns the unique
    number of elements each execution has returned in its picks.

    :param num_diff_items: number of different items that the bag will contain
    :param num_samples: number of samples to take from the bag
    :param num_of_series: number of times the sampling with replacement will be executed
    :return: array with the number of unique elements of each executio of sampling with replacement
    """
    unique_elems: np.ndarray = np.zeros(num_of_series)
    for i in range(num_of_series):
        picks = sampling_with_replacement(num_diff_items, num_samples)
        unique_elems[i] = len(np.unique(picks))
    return unique_elems


def count_uniques_in_series(unique_elements: np.ndarray) -> Dict[int, int]:
    """
    Method that returns a dictionary with the item and the number of occurrences the item has in
    a given array.

    :param unique_elements: numpy array with integers
    :return: dictionary with the item and the number of occurrences
    """
    unique, counts = np.unique(unique_elements, return_counts=True)
    return dict(zip(unique, counts))


if __name__ == '__main__':
    sns.set()
    elements = execute_series_of_sampling(num_diff_items=100, num_samples=500, num_of_series=100)
    ax = sns.distplot(elements)
    plt.show()
