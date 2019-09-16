import numpy as np
from back_propagation_implemention_layers.functions import *
class Affine:

    x = None
    w = None
    b = None

    gradients = None

    def __init__(self, w, b):
        self.w = w
        self.b = b

    def forward(self, x):
        self.x = x
        z0 = np.dot(self.x, self.w)
        z = z0 + self.b
        return z

    def back(self,dout):

        db = np.sum(dout,axis=0)
        dw = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.w.T)

        self.gradients = {}
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
        t = 1 / (1 + np.exp(-x))
        self.t = t
        return self.t

    def back(self, dout):
        dx = ((1-self.t)*self.t)*dout
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

    def back(self, dout):
        dx = (self.y - self.t)*dout / self.y.shape[0]
        self.dx = dx
        return self.dx
