import numpy as np
from back_propagation_implemention_layers.functions import *
class Affine:

    x = None
    w = None
    b = None

    gradients = {}

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        self.x = x
        z0 = np.dot(self.x, self.w)
        z = z0 + self.b
        return z

    def back(self,x,dout):
        self.forward(x)

        db = dout
        dw = np.dot(x.T,dout)
        dx = np.dot(dout,self.w.T)

        self.gradients['w'] = dw
        self.gradients['x'] = dx
        self.gradients['b'] = db
        return dx


class Sigmoid:

    x = None
    t = None

    dx = None


    def forward(self,x):
        self.x = x
        t =  1 / (1 + np.exp(-x))
        self.t = t
        return self.t

    def back(self,x,dout):
        self.forward(x)
        dx = (1-self.t)*self.t+dout
        self.dx = dx
        return dx

class SoftMaxCrossEntropyError:
    x = None
    t = None
    y = None
    dx = None

    def forward(self,x,t):
        self.x = x
        self.t = t
        y = softmax(self.x)
        self.y = y
        result = cross_entropy_error(self.y, self.t)
        return result

    def back(self):
        dx = self.y - self.t
        self.dx = dx
        return self.dx









