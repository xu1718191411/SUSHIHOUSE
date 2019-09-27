from conv_pool_implement.functions import *

class Relu:

    dx = None
    x = None
    def forward(self,x):
        mask = x < 0
        result = np.copy(x)
        result[mask] = 0
        self.x = x
        return result

    def back(self,dout):
        self.dx = np.copy(self.x)
        mask0 = self.x < 0
        mask1 = self.x >= 0
        self.dx[mask0] = 0
        self.dx[mask1] = 1
        return self.dx * dout


class Conv:

    x = None

    w = None #filter
    b = None
    dw = None
    db = None

    col = None
    colW = None

    filterSizeH = None
    filterSizeW = None
    stride = None
    padding = None


    def __init__(self,w,b,filterSizeW,filterSizeH,stride,padding):
        self.w = w
        self.b = b
        self.filterSizeH = filterSizeH
        self.filterSizeW = filterSizeW
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        self.x = x
        xNum = x.shape[0]
        xColorNum = x.shape[1]
        xHeight = x.shape[2]
        xWidth = x.shape[3]

        fNum = self.w.shape[0]
        fCNum = self.w.shape[1]
        fHeight = self.w.shape[2]
        fWidth = self.w.shape[3]

        col = im2col(x,self.filterSizeH,self.filterSizeW,self.stride,self.padding)
        colW = np.reshape(self.w,[fNum,-1]).T

        self.col = col
        self.colW = colW

        finalW = int(((xWidth + self.padding * 2) - self.filterSizeW) / self.stride + 1)
        finalH = int(((xHeight + self.padding * 2) - self.filterSizeH) / self.stride + 1)

        res = np.dot(col,colW) + self.b
        result = np.reshape(res,[xNum,finalH,finalW,-1]).transpose(0,3,1,2)

        return result

    def back(self,dout):
        fNum,fCNum,fHeight,fWidth = self.w.shape
        dout = dout.transpose(0,2,3,1)
        dout = np.reshape(dout,[-1,fNum])

        self.db = np.sum(dout,axis=0)
        self.dw = np.dot(self.col.T,dout)
        self.dw = self.dw.transpose(1,0)
        self.dw = np.reshape(self.dw,[fNum,fCNum,fHeight,fWidth])


        dcol = np.dot(dout,self.colW.T)
        dx = col2im(dcol,self.x.shape,fHeight,fWidth,self.stride,self.padding)
        return dx


class Pool:

    poolW = None
    poolH = None
    stride = None
    padding = None
    x = None
    argMax = None

    def __init__(self,poolW,poolH,stride,padding):
        self.poolW = poolW
        self.poolH = poolH
        self.stride = stride
        self.padding = padding


    def forward(self,x):
        self.x = x

        xNum,xCNum,xH,xW = x.shape
        finalH = int(((xH + 2*self.padding) - self.poolH)/self.stride + 1)
        finalW = int(((xW + 2*self.padding) - self.poolW)/self.stride + 1)

        col = im2col(x,self.poolH,self.poolW,self.stride,self.padding)
        res0 = np.reshape(col,[-1,self.poolH*self.poolW])
        res1 = np.max(res0,axis=1)
        self.argMax = np.argmax(res0,axis=1)

        res2 = np.reshape(res1,[xNum,finalH,finalW,-1]).transpose(0,3,1,2)
        return res2

    def back(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.poolH * self.poolW
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.argMax.size), self.argMax.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.poolH, self.poolW, self.stride, self.padding)

        return dx


class Affine:
    w = None
    b = None
    x = None
    dw = None
    db = None

    originalShape = None

    def __init__(self,w,b):
        self.w = w
        self.b = b

    def forward(self,x):

        self.originalShape = x.shape #目前还不明白为什么要加这个
        x = np.reshape(x,[self.originalShape[0],-1])
        self.x = x
        xNum = x.shape[0]
        x = np.reshape(x,[xNum,-1])
        res =  np.dot(x,self.w) + self.b
        result = sigmoid(res)
        return result

    def back(self,dout):
        dx = np.dot(dout,self.w.T)
        self.dw = np.dot(self.x.T,dout)

        self.db = np.sum(dout,axis=0)
        self.dx = np.reshape(dx,self.originalShape)# originalShape目前还不明白为什么要加这个
        return self.dx


class SoftmaxWithCrossEntropy:
    y = None
    t = None

    def forward(self,x,t):
        y = softmax(x)
        self.y = y
        self.t = t
        result = cross_entropy_error(y,t)
        return result

    def back(self,dout):
        return dout * (self.y - self.t) / self.t.shape[0]