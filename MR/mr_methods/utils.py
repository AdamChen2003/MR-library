import numpy as np


def pmin(x1, x2):
    """
    Computes the parallel minima between two vectors
    """
    arr = np.array([])
    for i in range(0, len(x1)):
        arr = np.append(arr, min(x1[i], x2[i]))

    return arr


def mad(data):
    """
    Computes median absolute deivation for provided data
    """
    return (abs(data-data.mean())).sum()/len(data)
