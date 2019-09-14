import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if(x.ndim == 1):
        max = np.max(x)
        x = x - max
        return np.exp(x) / np.sum(np.exp(x))
    else:
        arr = x.T
        max = np.max(arr,axis=0)
        arr = arr - max
        res = np.exp(arr) / np.sum(np.exp(arr),axis=0)
        result = res.T
        return result

def cross_entropy_error(x,t):
    delta = 1e-7
    return np.sum(-1 * t * np.log(x+delta)) / x.shape[0]