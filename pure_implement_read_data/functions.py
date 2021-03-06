import numpy as np

# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# def softmax(x):
#     if x.ndim == 1:
#         max = np.max(x)
#         x = x - max
#         return np.exp(x) / np.sum(np.exp(x))
#     else:
#         x = x.T
#         max = np.max(x,axis=0)
#         x = x - max
#         res = np.exp(x) / np.sum(np.exp(x),axis=0)
#         return res.T
#
# def cross_entropy_error(x,t):
#     delta = 1e-4
#     return np.sum(-1 * t * np.log(x + delta)) / x.shape[0]
#



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if(x.ndim == 1):
        x = np.reshape(x,[1,x.size])

    dim = x.shape[0]
    if(dim == 1):
        max = np.max(x)
        x = x - max
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x = x.T
        max = np.max(x,axis=0)
        x = x - max
        return (np.exp(x) / np.sum(np.exp(x),axis=0)).T

def cross_entropy_error(x, t):
    delta = 1e-4
    return np.sum(-1 * t * np.log(x + delta))/x.shape[0]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    if(x.ndim == 1):
        x = np.reshape(x,[1,x.size])

    dim = x.shape[0]
    if(dim == 1):
        max = np.max(x)
        x = x - max
        return np.exp(x) / np.sum(np.exp(x))
    else:
        x = x.T
        max = np.max(x,axis=0)
        x = x - max
        return (np.exp(x) / np.sum(np.exp(x),axis=0)).T

def crossEntropyError(x, t):
    delta = 1e-7
    return np.sum(-1 * t * np.log(x + delta))/x.shape[0]



def numerical_gradients(x,f):
    arr = np.nditer(x,flags=["multi_index"],op_flags=["readwrite"])
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