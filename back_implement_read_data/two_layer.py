import numpy as np
from back_implement_read_data.functions import *

class TwoLayer:
    inputSize = None
    hiddenSize = None
    outputSize = None

    z0 = None
    z1 = None
    z2 = None
    z3 = None
    z4 = None


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


    def back(self,x,t,dz=1):
        z0 = np.dot(x,self.params['w1'])
        z1 = z0 + self.params['b1']
        z2 = sigmoid(z1)
        z3 = np.dot(z2,self.params['w2'])
        z4 = z3 + self.params['b2']
        y = softmax(z4)
        g0 = ((y - t)/x.shape[0]) * dz
        gb2 = np.sum(g0)
        gw2 = np.dot(z2.T,g0)
        g1 = np.dot(g0, self.params['w2'].T)

        g2 = z2*(1-z2)*g1
        gb1 = np.sum(g2)

        gw1 = np.dot(x.T, g2)
        gradients = {}
        gradients['w1'] = gw1
        gradients['b1'] = gb1
        gradients['w2'] = gw2
        gradients['b2'] = gb2
        return gradients

    def calculate_numerical_gradients(self,x,t):
        f = lambda p:self.loss(x,t)
        gradients = {}
        gradients['w1'] = numerical_gradients(self.params['w1'], f)
        gradients['b1'] = numerical_gradients(self.params['b1'], f)
        gradients['w2'] = numerical_gradients(self.params['w2'], f)
        gradients['b2'] = numerical_gradients(self.params['b2'], f)
        return gradients





