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
    # numerical_gradients = cnnLayer.numerical_gradient(trainX, trainT)

    cnnLayer.params['w1'] = cnnLayer.params['w1'] - LEARNING_RATE * gradients['dw1']
    cnnLayer.params['b1'] = cnnLayer.params['b1'] - LEARNING_RATE * gradients['db1']
    cnnLayer.params['w2'] = cnnLayer.params['w2'] - LEARNING_RATE * gradients['dw2']
    cnnLayer.params['b2'] = cnnLayer.params['b2'] - LEARNING_RATE * gradients['db2']
    cnnLayer.params['w3'] = cnnLayer.params['w3'] - LEARNING_RATE * gradients['dw3']
    cnnLayer.params['b3'] = cnnLayer.params['b3'] - LEARNING_RATE * gradients['db3']
    print(loss)
