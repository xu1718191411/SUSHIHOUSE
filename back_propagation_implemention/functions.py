import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    if (x.ndim == 1):
        max = np.max(x)
        x = x - max
        return np.exp(x) / np.sum(np.exp(x))
    else:
        arr = x.T
        max = np.max(arr, axis=0)
        arr = arr - max
        result = np.exp(arr) / np.sum(np.exp(arr), axis=0)
        return result.T


def cross_entropy_error(x, t):
    delta = 1e-7
    res = -1 * t * np.log(x + delta)
    total = np.sum(res)
    return total / x.shape[0]


def numerical_gradients(x,f):
    arr = np.nditer(x,flags=['multi_index'],op_flags=['readwrite'])
    delta = 1e-4
    result = np.zeros_like(x)
    while not arr.finished:

        index = arr.multi_index
        value = x[index]

        x[index] = value - delta
        s1 = f(x)

        x[index] = value + delta
        s2 = f(x)

        res = (s2 - s1) / (2*delta)
        result[index] = res

        x[index] = value


        arr.iternext()
    return result