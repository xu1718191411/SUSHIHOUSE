from back_propagation_implemention.init_data import *
from back_propagation_implemention.two_layer import  *

trainImg,trainLabel,testImg,testLabel = init_data()


inputSize = trainImg.shape[1]
HIDDEN_SIZE = 50
OUTPUT_SIZE = 10
BATCH_SIZE = 100
LEARNING_RATE = 0.1
LEARNING_NUM = 10000

layer = TwoLayer(inputSize, HIDDEN_SIZE, OUTPUT_SIZE)

for i in range(LEARNING_NUM):

    mask = np.random.choice(trainImg.shape[0], BATCH_SIZE)
    input = trainImg[mask]
    inputT = trainLabel[mask]


    loss = layer.loss(input, inputT)
    gradients = layer.back(input, inputT, 1)

    layer.params['w1'] = layer.params['w1'] - LEARNING_RATE * gradients['w1']
    layer.params['b1'] = layer.params['b1'] - LEARNING_RATE * gradients['b1']
    layer.params['w2'] = layer.params['w2'] - LEARNING_RATE * gradients['w2']
    layer.params['b2'] = layer.params['b2'] - LEARNING_RATE * gradients['b2']




    if(i % BATCH_SIZE == 0):
        testMask = np.random.choice(testImg.shape[0], BATCH_SIZE)
        testX = testImg[testMask]
        testT = testLabel[testMask]
        accuracy = layer.accuracy(testX,testT)
        print("acc:",accuracy)