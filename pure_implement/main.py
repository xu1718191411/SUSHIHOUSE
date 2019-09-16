import numpy as np
from pure_implement.two_layer import TwoLayer
from pure_implement.functions import createMockTeacherData
DATA_DIM = 100
DATA_X_NUM = 28
DATA_Y_NUM = 28
HIDDEN_SIZE = 30
OUTPUT_SIZE = 10
MIN_VALUE = 0
MAX_VALUE = 255
LEARNING_NUM = 10000
LEARNING_RATE = 0.01
BATCH_SIZE = 10

def initData(dataDim,dataXNum,dataYNum,minValue,maxValue):
    inputData = np.random.randint(minValue,maxValue,[dataDim,dataXNum,dataYNum])/maxValue
    return np.reshape(inputData,[DATA_DIM,DATA_X_NUM*DATA_Y_NUM])

input = initData(DATA_DIM,DATA_X_NUM,DATA_Y_NUM,MIN_VALUE,MAX_VALUE)

layer = TwoLayer(DATA_X_NUM*DATA_Y_NUM,HIDDEN_SIZE,OUTPUT_SIZE)
result = layer.predict(input)

teacher = createMockTeacherData(DATA_DIM,OUTPUT_SIZE)

def train(train,train_label,layer):
    for i in range(LEARNING_NUM):
        mask = np.random.choice(DATA_DIM,BATCH_SIZE)
        trainBatch = train[mask]
        labelBatch = train_label[mask]
        loss = layer.forward(trainBatch, labelBatch)
        gradients = layer.numericalGradient(trainBatch,labelBatch)
        for key in gradients.keys():
            layer.params[key] = layer.params[key] - LEARNING_RATE * gradients[key]
        print(loss)
        acc = layer.accuracy(trainBatch,labelBatch)
        print("acc:",acc)


train(input,teacher,layer)