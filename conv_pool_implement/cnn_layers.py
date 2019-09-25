from conv_pool_implement.layers import *

class CNNLayers:
    params = {}
    layers = {}

    paramWeight = None
    imageSizeW = None
    imageSizeH = None
    convFilter = {}
    poolParams = {}

    hiddentSize = None
    outputSize = None

    def __init__(self):
        self.initParams()
        self.initLayers()
        pass

    def initParamsValues(self):
        self.paramWeight = 0.01
        self.convFilter['filterColorDim'] = 1
        self.convFilter['filterNum'] = 3
        self.convFilter['filterSizeH'] = 5
        self.convFilter['filterSizeW'] = 5
        self.convFilter['stride'] = 1
        self.convFilter['padding'] = 0

        self.poolParams['poolH'] = 4
        self.poolParams['poolW'] = 4
        self.poolParams['stride'] = 4
        self.poolParams['padding'] = 0

        self.imageSizeW = 28 ##输入的图片的原始宽度
        self.imageSizeH = 28 ##输入图片的原始高度

    def initParams(self):
        weight = self.paramWeight
        filterNum = self.convFilter['filterNum']
        filterColorDim = self.convFilter['filterColorDim']
        filterSizeH = self.convFilter['filterSizeH']
        filterSizeW = self.convFilter['filterSizeW']

        self.params['w1'] = weight * np.random.randn(filterNum, filterColorDim, filterSizeH, filterSizeW)
        self.params['b1'] = weight * np.random.randn(filterNum)

        finalConvW = int(((self.imageSizeW + self.convFilter['padding'] * 2 - self.convFilter['filterSizeW'])/self.convFilter['stride']) + 1)
        finalConvH = int(((self.imageSizeH + self.convFilter['padding'] * 2 - self.convFilter['filterSizeH'])/self.convFilter['stride']) + 1)

        finalPoolW = (int)(((finalConvW + self.poolParams['padding'] * 2 - self.poolParams['poolW'])/self.poolParams['poolW']) + 1)
        finalPoolH = (int)(((finalConvH + self.poolParams['padding'] * 2 - self.poolParams['poolH'])/self.poolParams['poolH']) + 1)

        self.params['w2'] = np.random.randn(filterNum*finalPoolW*finalPoolH,self.hiddentSize)
        self.params['b2']

    def initLayers(self):
        self.layers['conv'] = Conv(self.params['w1'],self.params['b1'],self.convFilter['filterSizeH'],self.convFilter['filterSizeW'],self.convFilter['stride'],self.convFilter['padding'])
        self.layers['relu'] = Relu()
        self.layers['pool'] = Pool()
        self.layers['affine1'] = Affine()
        self.layers['relu'] = Relu()
        self.layers['affine2'] = Affine()
        self.lastLayer = SoftmaxWithCrossEntropy()

    def forward(self,x):
        pass

    def backward(self):
        pass


