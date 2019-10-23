import numpy as np
from conv_pool_optimizer_implement.functions import *
class Relu:
    x = None
    mask = None
    def forward(self,x):
        self.x = x
        mask = x <= 0
        self.mask = mask
        result = np.copy(x)
        result[mask] = 0
        return result

    def backward(self,dout):
        dout[self.mask] = 0
        return dout


class Affine:
    x = None
    w = None
    b = None

    dx = None
    dw = None
    db = None

    def __init__(self,w,b):
        self.w = w
        self.b = b

    def forward(self,x):
        self.x = x
        z0 = np.dot(x,self.w)
        z1 = z0 + self.b
        return z1

    def backward(self,dout):
        self.dx = np.dot(dout,self.w.T)
        self.dw = np.dot(self.x.T,dout)
        self.db = np.sum(dout,axis=0) #不用除以维度，因为axix=0了,这个和下面的结果是一样的
        db = np.sum(dout,axis=0)
        return self.dx


class CrossEntropyError:
    x = None
    t = None
    y = None

    def forward(self,x,t):
        self.x = x
        self.t = t
        delta = 1e-7
        y = -1 * np.log(x+delta) * t
        self.y = y
        result = np.sum(y) / x.shape[0]
        return result

    def backward(self,dout):
        result = (self.y - self.t) * dout
        return result / self.x.shape[0] #记住这里要除以batch_size



class Conv:

    x = None
    inputNum = None
    inputColorChannel = None
    inputX = None
    inputH = None
    filter = None
    filterX = None
    filterH = None

    stride = None

    def __init__(self,x,filterX,filterH,stride,padding):
        self.x = x
        self.inputX = x.shape[2]
        self.inputH = x.shape[3]
        self.inputNum = x.shape[0]
        self.inputColorChannel = x.shape[1]
        self.filterX = filterX
        self.filterH = filterH
        self.stride = stride
        self.padding = padding
        self.initFilter()

    def initFilter(self):
        self.filter = np.random.randn(self.inputColorChannel,self.filterX,self.filterH)

    def forward(self):
        cols = im2col(self.x,self.filterH,self.filterX,self.stride,self.padding)
        filter = self.filter.reshape([self.inputColorChannel * self.filterX * self.filterH,-1])
        res = np.dot(cols,filter)

        finalXNum = int(((self.inputX + 2*self.padding - self.filterX) / self.stride) + 1)
        finalHNum = int(((self.inputH + 2*self.padding - self.filterH) / self.stride) + 1)

        res = np.reshape(res,[self.inputNum,1,finalXNum,finalHNum])
        return res

    def backward(self,dout):
        pass

