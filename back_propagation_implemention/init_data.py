import os
import pickle
import numpy as np

def one_hot(x):
    result = np.zeros([x.shape[0],10])
    for index in range(len(x)):
        result[index][x[index]] = 1
    return result

def init_data():
    path = os.path.dirname(os.path.dirname(__file__)) + "/data/mnist.pkl"
    with open(path,'rb') as f:
        data = pickle.load(f)
        trainImg = data['train_img']
        trainLabel = one_hot(data['train_label'])
        testImg = data['test_img']
        testLabel = one_hot(data['test_label'])
        return trainImg,trainLabel,testImg,testLabel

