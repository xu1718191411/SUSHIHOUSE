import numpy as np
from conv_pool_implement.layers import *

from conv_pool_implement.init_data import *

trainImg,trainLabel,testImg,testLabel = init_data()

BATCH_SIZE = 30

batchMask = np.random.choice(trainImg.shape[0],BATCH_SIZE);

trainX = trainImg[batchMask]
filterColorDim = 1
filterNum = 1
filterSizeH = 5
filterSizeW = 5
stride = 1
padding = 0
w = np.random.randn(filterNum,filterColorDim,filterSizeW,filterSizeH)

conv = Conv(w,filterSizeH,filterSizeW,stride,padding)

conv.forward(trainX)
