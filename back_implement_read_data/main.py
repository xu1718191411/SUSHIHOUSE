from back_implement_read_data.init_data import init_data
from back_implement_read_data.functions import *
from back_implement_read_data.two_layer import *


trainImg, trainLabel, testImg, testLabel = init_data()

BATCH_SIZE = 100
HIDDEN_SIZE = 30
OUTPUT_SIZE = 10


layer = TwoLayer(trainImg.shape[1],HIDDEN_SIZE,OUTPUT_SIZE)

mask = np.random.choice(trainImg.shape[0],BATCH_SIZE)

input = trainImg[mask]

result = layer.predict(input)
print(1)
