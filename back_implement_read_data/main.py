from back_implement_read_data.init_data import init_data
from back_implement_read_data.functions import *


trainImg, trainLabel, testImg, testLabel = init_data()

arr = np.random.randn(12,10)

res = sigmoid(arr)
result = softmax(res)
print(1)


