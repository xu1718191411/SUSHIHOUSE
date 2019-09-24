import os
import pickle
import numpy as np

def init_data():
    path = os.path.dirname(os.path.dirname(__file__)) + "/data/mnist.pkl";
    with open(path,'rb') as f:
        data = pickle.load(f)
        trainImg = formart_to_multi_dimensions(data['train_img'])

        trainLabel = data['train_label']
        data_to_one_hot(trainLabel)
        testImg = formart_to_multi_dimensions(data['test_img'])
        testLabel = data['test_label']
        return trainImg,trainLabel,testImg,testLabel

def formart_to_multi_dimensions(x):
    return np.reshape(x,[-1,1,28,28])


def data_to_one_hot(x):
    result = np.zeros([x.shape[0],10])
    for index,value in enumerate(x):
        result[index][value] = 1
    return result
