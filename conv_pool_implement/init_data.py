import os
import pickle
import numpy as np

def init_data():
    path = os.path.dirname(os.path.dirname(__file__)) + "/data/mnist.pkl";
    with open(path,'rb') as f:
        data = pickle.load(f)
        trainImg = formart_to_multi_dimensions(data['train_img'])
        trainLabel = data['train_label']
        testImg = formart_to_multi_dimensions(data['test_img'])
        testLabel = data['test_label']
        return trainImg,trainLabel,testImg,testLabel

def formart_to_multi_dimensions(x):
    return np.reshape(x,[-1,1,28,28])
