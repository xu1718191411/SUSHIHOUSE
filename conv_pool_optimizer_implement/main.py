from conv_pool_optimizer_implement.init_data import *
from conv_pool_optimizer_implement.layers import *

trainImg,trainLabel,testImg,testLabel = init_data()

BATCH_SIZE = 100


trainImg = np.random.randn(100,3,28,28)

mask = np.random.choice(trainImg.shape[0],BATCH_SIZE)
x = trainImg[mask]
conv = Conv(x,3, 3, 3, 1, 0)

res = conv.forward()

pool = Pool(3,3,1,0)
pool.forward(res)
