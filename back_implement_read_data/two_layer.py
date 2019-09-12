import numpy as np
from back_implement_read_data.functions import *

class TwoLayer:
    inputSize = None
    hiddenSize = None
    outputSize = None

    params = {}

    def __init__(self,inputSize,hiddenSize,outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.initParams(0.01)

    def initParams(self,weight=0.01):
        self.params['w1'] = weight * np.random.randn(self.inputSize,self.hiddenSize)
        self.params['b1'] = np.zeros(self.hiddenSize)
        self.params['w2'] = weight * np.random.randn(self.hiddenSize,self.outputSize)
        self.params['b2'] = np.zeros(self.outputSize)

    def predict(self,x):
        z0 = np.dot(x,self.params['w1'])
        z1 = z0 + self.params['b1']
        z2 = sigmoid(z1)
        z3 = np.dot(z2,self.params['w2'])
        z4 = z3 + self.params['b2']
        y = softmax(z4)
        return y

    def loss(self,x,t):
        res = self.predict(x)
        loss = cross_entropy_error(res,t)
        return loss



