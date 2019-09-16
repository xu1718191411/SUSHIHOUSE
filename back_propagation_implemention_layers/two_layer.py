from back_propagation_implemention_layers.layers import *
from collections import OrderedDict

class TwoLayer:

    inputSize = None
    hiddenSize = None
    outputSize = None
    params = {}
    layers = OrderedDict()
    lastLayer = None

    def __init__(self,inputSize,hiddenSize,outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.init_params(0.01)
        self.init_layers()

    def init_params(self,weight=0.01):
        self.params['w1'] = weight * np.random.randn(self.inputSize,self.hiddenSize)
        self.params['b1'] = np.random.rand(1,self.hiddenSize)
        self.params['w2'] = weight * np.random.randn(self.hiddenSize,self.outputSize)
        self.params['b2'] = np.random.randn(1,self.outputSize)

    def init_layers(self):
        self.layers['affine1'] = Affine(self.params['w1'],self.params['b1'])
        self.layers['sigmoid'] = Sigmoid()
        self.layers['affine2'] = Affine(self.params['w2'],self.params['b2'])
        self.layers.lastLayer = SoftMaxCrossEntropyError()

    def predict(self,x):
        for key in self.layers.keys():
            x = self.layers[key].forward(x)
        return x

    def forward(self, x, t):
        x = self.predict(x)
        result = self.layers.lastLayer.forward(x,t)
        return result

    def gradients(self,x,t,dout):
        self.forward(x,t)
        dout = dout * self.layers.lastLayer.back(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.back(dout)

        gradients = {}
        gradients['w1'] = self.layers['affine1'].gradients['w']
        gradients['b1'] = self.layers['affine1'].gradients['b']
        gradients['w2'] = self.layers['affine2'].gradients['w']
        gradients['b2'] = self.layers['affine2'].gradients['b']
        return gradients


    def accuracy(self,testX,testT):
        xResult = self.predict(testX)
        xArgResult = np.argmax(xResult,axis=1)
        tArgResult = np.argmax(testT,axis=1)

        equalNum = np.sum(xArgResult == tArgResult)
        rate = equalNum / testX.shape[0]
        return rate

