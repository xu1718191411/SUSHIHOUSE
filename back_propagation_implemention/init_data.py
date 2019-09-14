import os
import pickle
import numpy as np

def one_hot(x):
    result = np.zeros([x.shape[0],10])
    for index in range(len(x)):
        result[index][x[index]] = 1
    return result

def normolization(x):
    x = x / 255.0
    return x

def init_data():
    path = os.path.dirname(os.path.dirname(__file__)) + "/data/mnist.pkl"
    with open(path,'rb') as f:
        data = pickle.load(f)
        trainImg = normolization(data['train_img'])
        trainLabel = one_hot(data['train_label'])
        testImg = normolization(data['test_img'])
        testLabel = one_hot(data['test_label'])
        return trainImg,trainLabel,testImg,testLabel

