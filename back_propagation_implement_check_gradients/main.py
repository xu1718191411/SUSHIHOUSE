from back_propagation_implement_check_gradients.init_data import init_data
from back_propagation_implement_check_gradients.two_layer import *

trainImg, trainLabel, testImg, testLabel = init_data()

BATCH_SIZE = 100
HIDDEN_SIZE = 30
OUTPUT_SIZE = 10


def checkGradients(trainImg, trainLabel, batchSize, hiddenSize, outputSize):
    layer = TwoLayer(trainImg.shape[1], hiddenSize, outputSize)
    mask = np.random.choice(trainImg.shape[0], batchSize)
    input = trainImg[mask]
    teacher = trainLabel[mask]
    gradients = layer.back(input, teacher, 1)
    numericalGradients = layer.calculate_numerical_gradients(input, teacher)
    for key in gradients.keys():
        gradient = gradients[key]
        res = np.abs(gradients[key] - numericalGradients[key])
        result = np.average(res)
        print(key, ":", result)


checkGradients(trainImg, trainLabel, BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
