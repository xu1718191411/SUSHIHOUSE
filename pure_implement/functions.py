import numpy as np

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
    delta = 1e-4
    return np.sum(-1 * t * np.log(x + delta))/x.shape[0]

def createMockTeacherData(dim,num):
    result = np.zeros([dim,num])
    for key,index in enumerate(result):
        random = np.random.randint(0, num - 1)
        result[key][random] = 1
    return result

def calculateNumericalGradients(x, f):
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

        result[index] = (s2 - s1) / (2*delta)
        x[index] = value
        arr.iternext()

    return result