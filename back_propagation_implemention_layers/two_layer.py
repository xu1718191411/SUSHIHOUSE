from back_propagation_implemention_layers.layers import *
from collections import OrderedDict

class TwoLayer:

    inputSize = None
    hiddenSize = None
    outputSize = None
    params = {}
    layers = OrderedDict()

    def __init__(self,inputSize,hiddenSize,outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize
        self.init_params(0.01)
        self.init_layers()

    def init_params(self,weight=0.01):
        self.params['w1'] = weight * np.random.randn(self.inputSize,self.hiddenSize)
        self.params['b1'] = np.zeros(self.hiddenSize)
        self.params['w2'] = weight * np.random.randn(self.hiddenSize,self.outputSize)
        self.params['b2'] = np.zeros(self.outputSize)

    def init_layers(self):
        self.layers['affine1'] = Affine(self.params['w1'],self.params['b1'])
        self.layers['sigmoid'] = Sigmoid()
        self.layers['affine2'] = Affine(self.params['w2'],self.params['b2'])
        self.layers.lastLayer = SoftMaxCrossEntropyError()

    def predict(self,x):
        for key in self.layers.keys():
            x = self.layers[key].forward(x)
        return x



