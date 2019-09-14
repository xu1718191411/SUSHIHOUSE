from back_propagation_implemention_layers.data_init import *
from back_propagation_implemention_layers.two_layer import *

trainImg,trainLabel,testImg,testLabel = data_init()


HIDDEN_SIZE = 30
OUTPUT_SIZE = 10
BATCH_SIZE = 100

layer = TwoLayer(trainImg.shape[1],HIDDEN_SIZE,OUTPUT_SIZE)

mask = np.random.choice(trainImg.shape[0],BATCH_SIZE)

x = trainImg[mask]
out = layer.predict(x)



