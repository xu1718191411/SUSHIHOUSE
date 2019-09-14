import numpy as np
from back_propagation_implemention.functions import *

class TwoLayer:

    inputSize = None
    hiddenSize = None
    outputSize = None
    params = {}
    z0 = None
    z1 = None
    z2 = None
    z3 = None
    z4 = None
    y = None

    def __init__(self,inputSize,hiddenSize,outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.params_init(0.01)

    def params_init(self,weight=0.01):
        self.params = {}
        self.params['w1'] = weight * np.random.randn(self.inputSize,self.hiddenSize)
        self.params['b1'] =  np.zeros(self.hiddenSize)
        self.params['w2'] = weight * np.random.randn(self.hiddenSize,self.outputSize)
        self.params['b2'] = np.zeros(self.outputSize)

    def predict(self, x):
        self.z0 = np.dot(x, self.params['w1'])
        self.z1 = self.z0 + self.params['b1']
        self.z2 = sigmoid(self.z1)
        self.z3 = np.dot(self.z2, self.params['w2'])
        self.z4 = self.z3 + self.params['b2']
        self.y = softmax(self.z4)
        return self.y

    def loss(self,x,t):
        result = self.predict(x)
        loss = cross_entropy_error(result,t)
        return loss

    def back(self,x,t,dout):
        g0 = ((self.y - t)/ x.shape[0]) * dout
        gb2 = np.sum(g0,axis=0)
        gw2 = np.dot(self.z2.T,g0)
        g1 = np.dot(g0,self.params['w2'].T)
        g2 = self.z2*(1 - self.z2)*g1
        gb1 = np.sum(g2,axis=0)
        gw1 = np.dot(x.T,g2)

        gradients = {}

        gradients['w1'] = gw1
        gradients['b1'] = gb1
        gradients['w2'] = gw2
        gradients['b2'] = gb2
        return gradients

    def calculate_numerical_gradients(self,x,t):
        f = lambda params:self.loss(x,t)
        gradients = {}
        gradients['w1'] = numerical_gradients(self.params['w1'], f)
        gradients['b1'] = numerical_gradients(self.params['b1'], f)
        gradients['w2'] = numerical_gradients(self.params['w2'], f)
        gradients['b2'] = numerical_gradients(self.params['b2'], f)
        return gradients

    def accuracy(self,x,t):
        result = self.predict(x)
        argX = np.argmax(result,axis=1)
        argT = np.argmax(t,axis=1)

        equalNum = np.sum(argX == argT)
        result = equalNum / argX.shape[0]
        return result
