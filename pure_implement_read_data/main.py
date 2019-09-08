from pure_implement_read_data.data import *
from pure_implement_read_data.two_layer import TwoLayer

trainImg,trainLabel,testImg,testLabel = initData()

INPUT_DIM = trainImg.shape[0]
INPUT_SIZE = trainImg.shape[1]
HIDDEN_SIZE = 30
OUTPUT_SIZE = 10
LEARNING_NUM = 1000
BATCH_SIZE = 100
LEARNING_RATE = 0.01


for i in range(LEARNING_NUM):
    sizeMask = np.random.choice(INPUT_DIM,BATCH_SIZE)
    train = trainImg[sizeMask]
    trainTeacher = trainLabel[sizeMask]
    layer = TwoLayer(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    gradients = layer.numerical_gradients(train,trainTeacher)
    for key in gradients.keys():
        layer.params[key] = layer.params[key] - LEARNING_RATE * gradients[key]

    loss = layer.loss(train,trainTeacher)
    print(loss)

    frequent = 10

    if (i % frequent == 0):
        trainAcc = layer.accuracy(train,trainTeacher)
        print("trainAcc:",trainAcc)









