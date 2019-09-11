import numpy as np
import pickle
import os


def init_data():
    file = os.path.dirname(os.path.dirname(__file__))
    dataPath = file + "/data/mnist.pkl"

    with open(dataPath, 'rb') as f:
        data = pickle.load(f)
        trainImg = data['train_img']
        trainLabel = data['train_label']
        testImg = data['test_img']
        testLabel = data['test_label']
        return trainImg, trainLabel, testImg, testLabel


