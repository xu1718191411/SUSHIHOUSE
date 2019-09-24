import numpy as np
from conv_pool_implement.layers import *
from conv_pool_implement.init_data import *
trainImg,trainLabel,testImg,testLabel = init_data()


########################################################
BATCH_SIZE = 30
weight = 0.01
batchMask = np.random.choice(trainImg.shape[0],BATCH_SIZE)
trainX = trainImg[batchMask]
testX = trainLabel[batchMask]
filterColorDim = 1
filterNum = 3
filterSizeH = 5
filterSizeW = 5
stride = 1
padding = 0
w = weight * np.random.randn(filterNum,filterColorDim,filterSizeH,filterSizeW)
b = weight * np.random.randn(filterNum)
conv = Conv(w,b,filterSizeH,filterSizeW,stride,padding)
result1 = conv.forward(trainX)

#########################################################
relu = Relu()
result2 = relu.forward(result1)

########################################################
poolH = 4
poolW = 4
stride = 4
padding = 0

pool = Pool(poolH,poolW,stride,padding)

result3 = pool.forward(result2)


#######################################################

result3H = result3.shape[2]
result3W = result3.shape[3]
s = filterNum * result3H * result3W
hiddenSize = 100
affine1W = weight * np.random.randn(s,hiddenSize)
affine1B = weight * np.random.randn(hiddenSize)
affine1 = Affine(affine1W,affine1B)

result4 = affine1.forward(result3)


######################################################

result5 = relu.forward(result4)


######################################################

outputSize = 10
affine2W = weight * np.random.randn(result5.shape[1],outputSize)
affine2B = weight * np.random.randn(outputSize)
affine2 = Affine(affine2W,affine2B)

result6 = affine2.forward(result5)

####################################################

softmaxWithCrossEntropy = SoftmaxWithCrossEntropy()
loss = softmaxWithCrossEntropy.forward(result6,testX)

print("loss:",loss)