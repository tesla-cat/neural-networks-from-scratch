import numpy as np
from a_loss import mse_loss, compare

class Relu:
    def forward(s, x):
        s.x = x
        y = np.where(x < 0, 0, x)
        return y
    
    def backward(s, de_dy, *args):
        de_dx = de_dy * np.where(s.x < 0, 0, 1)
        return de_dx

#=======================================

import torch
import torch.nn as nn

def test():
    x = torch.rand((3, 3), requires_grad=True) 
    t = torch.ones_like(x)

    y = nn.ReLU()(x-0.5)
    err = nn.MSELoss()(y, t)
    err.backward()
    de_dx = x.grad

    relu = Relu()
    x, t = x.detach().numpy(), t.detach().numpy()
    
    y = relu.forward(x-0.5)
    err2, de_dx2 = mse_loss(y, t)
    de_dx2 = relu.backward(de_dx2)

    compare('err', err, err2)
    compare('de_dx', de_dx, de_dx2)

if __name__=='__main__':
    test()
