import numpy as np
from conv_pool_implement.layers import *

from conv_pool_implement.init_data import *

trainImg,trainLabel,testImg,testLabel = init_data()

BATCH_SIZE = 30

batchMask = np.random.choice(trainImg.shape[0],BATCH_SIZE);

trainX = trainImg[batchMask]
filterColorDim = 1
filterNum = 3
filterSizeH = 5
filterSizeW = 5
stride = 1
padding = 0
w = np.random.randn(filterNum,filterColorDim,filterSizeH,filterSizeW)
b = np.random.randn(filterNum)

conv = Conv(w,b,filterSizeH,filterSizeW,stride,padding)
result1 = conv.forward(trainX)


relu = Relu()
result2 = relu.forward(result1)


poolH = 4
poolW = 4
stride = 4
padding = 0

pool = Pool(poolH,poolW,stride,padding)

result3 = pool.forward(result2)