from conv_pool_implement.functions import *

class Relu:

    def forward(self,x):
        mask = x < 0
        result = np.copy(x)
        result[mask] = 0
        return result


class Conv:

    w = None #filter
    b = None
    filterSizeH = None
    filterSizeW = None
    stride = None
    padding = None

    def __init__(self,w,b,filterSizeH,filterSizeW,stride,padding):
        self.w = w
        self.b = b
        self.filterSizeH = filterSizeH
        self.filterSizeW = filterSizeW
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        xNum = x.shape[0]
        xColorNum = x.shape[1]
        xHeight = x.shape[2]
        xWidth = x.shape[3]

        fNum = self.w.shape[0]
        fCNum = self.w.shape[1]
        fHeight = self.w.shape[2]
        fWidth = self.w.shape[3]

        col = im2col(x,self.filterSizeH,self.filterSizeW,self.stride,self.padding)
        self.w = np.reshape(self.w,[fNum,-1]).T

        finalW = int(((xWidth + self.padding * 2) - self.filterSizeW) / self.stride + 1)
        finalH = int(((xHeight + self.padding * 2) - self.filterSizeH) / self.stride + 1)

        res = np.dot(col,self.w) + self.b
        result = np.reshape(res,[xNum,finalH,finalW,-1]).transpose(0,3,1,2)

        return result


class Pool:

    poolW = None
    poolH = None
    stride = None
    padding = None

    def __init__(self,poolW,poolH,stride,padding):
        self.poolW = poolW
        self.poolH = poolH
        self.stride = stride
        self.padding = padding


    def forward(self,x):
        xNum,xCNum,xH,xW = x.shape
        finalH = int(((xH + 2*self.padding) - self.poolH)/self.stride + 1)
        finalW = int(((xW + 2*self.padding) - self.poolW)/self.stride + 1)

        col = im2col(x,self.poolH,self.poolW,self.stride,self.padding)
        res0 = np.reshape(col,[-1,self.poolH*self.poolW])
        res1 = np.max(res0,axis=1)

        res2 = np.reshape(res1,[xNum,finalH,finalW,-1]).transpose(0,3,1,2)
        return res2


class Affine:
    w = None
    b = None

    def __init__(self,w,b):
        self.w = w
        self.b = b

    def forward(self,x):
        xNum = x.shape[0]
        x = np.reshape(x,[xNum,-1])
        res =  np.dot(x,self.w) + self.b
        result = sigmoid(res)
        return result


class SoftmaxWithCrossEntropy:

    def forward(self,x,t):
        softmaxResult = softmax(x)
        result = cross_entropy_error(softmaxResult,t)
        return result
