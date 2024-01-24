import numpy as np


def pmin(x1, x2):
    arr = np.array([])
    for i in range(0, len(x1)):
        arr = np.append(arr, min(x1[i], x2[i]))

    return arr
