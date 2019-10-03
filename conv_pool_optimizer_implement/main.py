from conv_pool_optimizer_implement.init_data import *
from conv_pool_optimizer_implement.layers import *

trainImg,trainLabel,testImg,testLabel = init_data()

BATCH_SIZE = 100


##################################################

input_size = 30
hidden_size_1 = 15
hidden_size_2 = 25
output_size = 10

x = np.random.randn(input_size,hidden_size_1)
w = np.random.randn(hidden_size_1,hidden_size_2)
b = np.random.randn(hidden_size_2)

affine1 = Affine(w,b)


w1 = np.random.randn(hidden_size_2,output_size)
b1 = np.random.randn(output_size)

affine2 = Affine(w1,b1)

result1 = affine1.forward(x)
result2 = affine2.forward(result1)

relu = Relu()

result = relu.forward(result2)
label = np.random.rand(input_size,output_size)

crossEntropyError = CrossEntropyError()
loss = crossEntropyError.forward(result,label)

print("loss:",loss)

dout = crossEntropyError.backward(1)

dout = relu.backward(dout)
dout = affine2.backward(dout)
dout = affine1.backward(dout)

s = np.zeros(5)
print(1)