import numpy as np
import pickle
import os

def init_data():
    dirname = os.path.dirname(os.path.dirname(__file__))

    path = dirname + "/data/mnist.pkl"
    with open(path,'rb') as f:
        data = pickle.load(f)
        trainImg = to_multi_dimensions(data['train_img'])
        trainLabel = to_one_hot(data['train_label'])
        testImg = to_multi_dimensions(data['test_img'])
        testLabel = to_one_hot(data['test_label'])
        return trainImg,trainLabel,testImg,testLabel

def to_multi_dimensions(x):
    x = np.reshape(x,[-1,1,28,28])
    return x

def to_one_hot(x):
    result = np.zeros([x.shape[0],10])
    for i in range(len(x)):
        result[i][x[i]] = 1
    return result