import numpy as np
from conv_pool_implement.init_data import *
from conv_pool_implement.cnn_layers import *

trainImg, trainLabel, testImg, testLabel = init_data()

BATCH_SIZE = 100
LEARNING_NUM = 1000
LEARNING_RATE = 0.01

cnnLayer = CNNLayers()

for i in range(LEARNING_NUM):

    train_mask = np.random.choice(trainImg.shape[0], BATCH_SIZE)
    test_mask = np.random.choice(testImg.shape[0], BATCH_SIZE)

    trainX = trainImg[train_mask]
    trainT = trainLabel[train_mask]

    # testX = testImg[test_mask]
    # testT = testLabel[test_mask]

    loss = cnnLayer.loss(trainX, trainT)
    gradients = cnnLayer.backward()

    for key in gradients.keys():
        cnnLayer.params[key] -= LEARNING_RATE * gradients[key]


    print(loss)
