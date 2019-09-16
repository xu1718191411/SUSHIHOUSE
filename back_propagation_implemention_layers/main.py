from back_propagation_implemention_layers.data_init import *
from back_propagation_implemention_layers.two_layer import *

trainImg,trainLabel,testImg,testLabel = data_init()

HIDDEN_SIZE = 30
OUTPUT_SIZE = 10
BATCH_SIZE = 100
LEARNING_NUM = 10000
LEARNING_RATE = 0.1

network = TwoLayer(trainImg.shape[1], HIDDEN_SIZE, OUTPUT_SIZE)

for index in range(LEARNING_NUM):

    mask = np.random.choice(trainImg.shape[0], BATCH_SIZE)
    x = trainImg[mask]
    t = trainLabel[mask]
    gradients = network.gradients(x, t, 1)

    for key in gradients.keys():
        network.params[key] -= LEARNING_RATE * gradients[key]

    if(index % BATCH_SIZE == 0):
        testMask = np.random.choice(testImg.shape[0],BATCH_SIZE)
        testImg = testImg[testMask]
        testLabel = testLabel[testMask]

        acc = network.accuracy(x,t)
        accTest = network.accuracy(testImg, testLabel)
        print("acc:", acc)
        print("acc test:", accTest)


