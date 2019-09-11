import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if(x.ndim == 1):
        max = np.max(x)
        x = x - max
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x = x.T
        max = np.max(x,axis=0)
        x = x - max
        res = np.exp(x) / np.sum(np.exp(x),axis=0)
        result = res.T
        return result

def cross_entropy_error(x,t):
    delta = 1e-4
    return np.sum(t * -1 * np.log(x + delta)) / x.shape[0]


