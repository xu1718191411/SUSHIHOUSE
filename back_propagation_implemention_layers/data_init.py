import os
import pickle
import numpy as np

def normolize(x):
    x = x / 255.0
    return x

def ont_hot(x):
    result = np.zeros([x.shape[0],10])

    for index in range(result.shape[0]):
        result[index][x[index]] = 1

    return result

def data_init():
    path = os.path.dirname(os.path.dirname(__file__)) + "/data/mnist.pkl"
    with open(path,'rb') as f:
        data = pickle.load(f)
        trainImg = normolize(data['train_img'])
        trainLabel = ont_hot(data['train_label'])
        testImg = normolize(data['test_img'])
        testLabel = ont_hot(data['test_label'])
        return trainImg,trainLabel,testImg,testLabel




