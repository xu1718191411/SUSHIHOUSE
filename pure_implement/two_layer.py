import numpy as np
from pure_implement.functions import *


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
        self.params['w1'] = weight * np.random.rand(self.inputSize,self.hiddenSize)
        self.params['b1'] = np.zeros([1,self.hiddenSize])
        self.params['w2'] = weight * np.random.rand(self.hiddenSize,self.outputSize)
        self.params['b2'] = np.zeros([1,self.outputSize])

    def predict(self,x):
        z0 = np.dot(x,self.params['w1'])
        z1 = z0 + self.params['b1']
        z2 = sigmoid(z1)
        z3 = np.dot(z2,self.params['w2'])
        z4 = z3 + self.params['b2']
        y = softmax(z4)
        return y

    def loss(self,x,t):
        result = self.predict(x)
        return crossEntropyError(result,t)

    def numericalGradient(self,x,t):

        f = lambda p:self.loss(x,t)
        self.numericalGradients['w1'] = calculateNumericalGradients(self.params['w1'],f)
        self.numericalGradients['w2'] = calculateNumericalGradients(self.params['w2'],f)
        self.numericalGradients['b1'] = calculateNumericalGradients(self.params['b1'], f)
        self.numericalGradients['b2'] = calculateNumericalGradients(self.params['b2'], f)
        return self.numericalGradients

    def accuracy(self,x, t):
        x = self.predict(x)
        if (x.ndim > 1):
            x = np.argmax(x, axis=1)
        if (t.ndim > 1):
            t = np.argmax(t,axis=1)

        num = np.sum((x == t))
        totalNum = x.shape[0]
        return num / totalNum


