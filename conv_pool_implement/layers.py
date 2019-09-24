from conv_pool_implement.functions import *

class Conv:

    w = None #filter
    filterSizeH = None
    filterSizeW = None
    stride = None
    padding = None

    def __init__(self,w,filterSizeH,filterSizeW,stride,padding):
        self.w = w
        self.filterSizeH = filterSizeH
        self.filterSizeW = filterSizeW
        self.stride = stride
        self.padding = padding

    def forward(self,x):
        print(1)
        col = im2col(x,self.filterSizeH,self.filterSizeW,self.stride,self.padding)
        print(2)