import os
import pickle
import numpy as np

def initData():

    dataFilPath = os.path.dirname(os.path.dirname(__file__)) + "/data/mnist.pkl"
    print(dataFilPath)
    with open(dataFilPath,'rb') as f:
        data = pickle.load(f)
        trainImg = data['train_img']
        trainLabel = data['train_label']
        testImg = data['test_img']
        testLabel = data['test_label']
        trainLabel = onehot(trainLabel)
        testLabel = onehot(testLabel)
    return trainImg,trainLabel,testImg,testLabel


def onehot(x):
    dim = x.shape[0]
    res = np.zeros([dim,10])

    for i in range(dim):
        maxarg = x[i]
        res[i][maxarg] = 1
    return res


