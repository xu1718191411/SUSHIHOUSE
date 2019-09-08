import numpy as np
from pure_implement_read_data.functions import *
#
#
class TwoLayer:
    inputSize = None
    hiddenSize = None
    outputSize = None
    params = {}
    numericalGradients = {}

    def __init__(self,inputSize,hiddenSize,outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.initParams(0.01)

    def initParams(self,weight=0.01):
        self.params['w1'] = weight * np.random.randn(self.inputSize,self.hiddenSize)
        self.params['b1'] = np.zeros([1,self.hiddenSize])
        self.params['w2'] = weight * np.random.randn(self.hiddenSize,self.outputSize)
        self.params['b2'] = np.zeros([1,self.outputSize])

    def predict(self,x):
        z0 = np.dot(x,self.params['w1'])
        z1 = z0 + self.params['b1']
        z2  = sigmoid(z1)
        z3 = np.dot(z2,self.params['w2'])
        z4 = z3 + self.params['b2']
        y = softmax(z4)
        return y

    def loss(self,x,t):
        res = self.predict(x)
        return cross_entropy_error(res,t)

    def accuracy(self,x,t):
        maxX = np.argmax(x,axis=1)
        maxT = np.argmax(t,axis=1)
        count = np.sum(maxX == maxT)
        x.shape[0]
        rate = count / x.shape[0]
        return rate

    def numerical_gradients(self,x,t):
        f = lambda p:self.loss(x,t)
        self.numericalGradients['w1'] = numerical_gradients(self.params['w1'], f)
        self.numericalGradients['b1'] = numerical_gradients(self.params['b1'], f)
        self.numericalGradients['w2'] = numerical_gradients(self.params['w2'], f)
        self.numericalGradients['b2'] = numerical_gradients(self.params['b2'], f)
        return self.numericalGradients