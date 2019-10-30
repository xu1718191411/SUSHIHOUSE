import numpy as np
from conv_pool_optimizer_implement.functions import *


class Relu:
    x = None
    mask = None

    def forward(self, x):
        self.x = x
        mask = x <= 0
        self.mask = mask
        result = np.copy(x)
        result[mask] = 0
        return result

    def backward(self, dout):
        dout[self.mask] = 0
        return dout


class Affine:
    x = None
    w = None
    b = None

    dx = None
    dw = None
    db = None

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        self.x = x
        z0 = np.dot(x, self.w)
        z1 = z0 + self.b
        return z1

    def backward(self, dout):
        self.dx = np.dot(dout, self.w.T)
        self.dw = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)  # 不用除以维度，因为axix=0了,这个和下面的结果是一样的
        db = np.sum(dout, axis=0)
        return self.dx


class CrossEntropyError:
    x = None
    t = None
    y = None

    def forward(self, x, t):
        self.x = x
        self.t = t
        delta = 1e-7
        y = -1 * np.log(x + delta) * t
        self.y = y
        result = np.sum(y) / x.shape[0]
        return result

    def backward(self, dout):
        result = (self.y - self.t) * dout
        return result / self.x.shape[0]  # 记住这里要除以batch_size


class Conv:
    x = None
    inputNum = None
    filterColorChannel = None
    inputX = None
    inputH = None
    filter = None
    filterW = None
    filterH = None
    f = None

    stride = None

    cols = None

    db = None
    df = None
    dx = None

    def __init__(self, x, filterX, filterH, filterNum, stride, padding):
        self.x = x
        self.inputX = x.shape[2]
        self.inputH = x.shape[3]
        self.inputNum = x.shape[0]
        self.filterColorChannel = x.shape[1]
        self.filterW = filterX
        self.filterH = filterH
        self.filterNum = filterNum
        self.stride = stride
        self.padding = padding
        self.initFilter()

    def initFilter(self):
        self.filter = np.random.randn(self.filterColorChannel, self.filterNum, self.filterW, self.filterH)

    def forward(self):
        cols = im2col(self.x, self.filterH, self.filterW, self.stride, self.padding)
        self.cols = cols
        f = self.filter.reshape([self.filterColorChannel * self.filterW * self.filterH, -1])
        res = np.dot(cols, f)
        self.f = f

        finalXNum = int(((self.inputX + 2 * self.padding - self.filterW) / self.stride) + 1)
        finalHNum = int(((self.inputH + 2 * self.padding - self.filterH) / self.stride) + 1)

        res = np.reshape(res, [self.inputNum, finalXNum, finalHNum, -1]).transpose(0, 3, 1, 2)

        return res

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        dout = np.reshape(dout, [-1, self.filterNum])

        self.db = np.sum(dout, axis=0)
        self.df = np.dot(self.cols.T, dout)
        self.df = self.df.transpose(1, 0)
        self.df = np.reshape(self.df, [self.filterNum, self.filterColorChannel, self.filterH, self.filterW])

        dcol = np.dot(dout, self.f.T)
        self.dx = col2im(dcol, self.x.shape, self.filterH, self.filterW, self.stride, self.padding)
        return self.dx
