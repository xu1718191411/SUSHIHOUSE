import numpy as np
from conv_pool_implement.init_data import *
from conv_pool_implement.cnn_layers import *

trainImg, trainLabel, testImg, testLabel = init_data()

BATCH_SIZE = 100
train_mask = np.random.choice(trainImg.shape[0],BATCH_SIZE)
test_mask = np.random.choice(testImg.shape[0],BATCH_SIZE)

trainX = trainImg[train_mask]
trainT = trainLabel[train_mask]

testX = testImg[test_mask]
testT = testLabel[test_mask]


cnnLayer = CNNLayers()

cnnLayer.loss(trainX,trainT)


cnnLayer.backward(1)