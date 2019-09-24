import os
import pickle
import numpy as np
def init_data():
    path = os.path.dirname(os.path.dirname(__file__)) + "/data/mnist.pkl";
    with open(path,'rb') as f:
        data = pickle.load(f)
        print(1)
        trainImg = formart(data['train_img'])
        trainLabel = formart(data['train_label'])
        testImg = formart(data['test_img'])
        testLabel = formart(data['test_label'])
        return trainImg,trainLabel,testImg,testLabel

def formart(x):
    return np.reshape(x,[-1,1,28,28])
