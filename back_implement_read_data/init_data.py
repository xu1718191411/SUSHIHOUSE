import numpy as np
import pickle
import os


def init_data():
    file = os.path.dirname(os.path.dirname(__file__))
    dataPath = file + "/data/mnist.pkl"

    with open(dataPath, 'rb') as f:
        data = pickle.load(f)
        trainImg = data['train_img']
        trainLabel = to_one_hot(data['train_label'])
        testImg = data['test_img']
        testLabel = to_one_hot(data['test_label'])
        return trainImg, trainLabel, testImg, testLabel


def to_one_hot(x):
    result = np.zeros([x.shape[0],10])
    for i in  range(x.shape[0]):
        result[i][x[i]] = 1
    return result


