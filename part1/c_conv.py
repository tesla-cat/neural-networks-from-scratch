import numpy as np
import numba
import numpy.random as npr
import torch
import torch.nn as nn
from c_full import Full

@numba.njit
def conv_forward(x, w, b, S=1):
    N, C1, H1, W1 = x.shape
    C2, C1, F, F = w.shape
    W2 = int((W1 - F)/S + 1)
    H2 = int((H1 - F)/S + 1)
    y = np.zeros((N, C2, H2, W2))
    for n in range(N):
        for c2 in range(C2):
            for h2 in range(H2):
                for w2 in range(W2):
                    i = h2*S; j = w2*S
                    y[n,c2,h2,w2] = np.sum(w[c2] * x[n,:,i:i+F,j:j+F]) + b[c2]
    return y

@numba.njit
def conv_backward(x, w, b, de_dy, S=1):
    N, C1, H1, W1 = x.shape
    C2, C1, F, F = w.shape
    W2 = int((W1 - F)/S + 1)
    H2 = int((H1 - F)/S + 1)
    de_dx = np.zeros_like(x)
    de_dw = np.zeros_like(w)
    de_db = np.zeros_like(b)
    for n in range(N):
        for c2 in range(C2):
            for h2 in range(H2):
                for w2 in range(W2):
                    i = h2*S; j = w2*S
                    de_dy_ = de_dy[n,c2,h2,w2]
                    de_dx[n,:,i:i+F,j:j+F] += de_dy_ * w[c2] 
                    de_dw[c2] += de_dy_ * x[n,:,i:i+F,j:j+F]
                    de_db[c2] += de_dy_
    return de_dx, de_dw, de_db

class Conv(Full):
    def __init__(s, C1, C2, F, S=1):
        s.S = S
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        k = np.sqrt( 1 / (C1 * F * F) )
        s.w = npr.uniform(-k, k, (C2, C1, F, F))
        s.b = npr.uniform(-k, k, (C2, ))
    
    def forward(s, x): 
        s.x = x
        return conv_forward(x, s.w, s.b, s.S)

    def backward(s, de_dy, lr):
        de_dx, s.de_dw, s.de_db = conv_backward(s.x, s.w, s.b, de_dy, s.S)
        s.w -= lr * s.de_dw
        s.b -= lr * s.de_db
        return de_dx

#=======================

from a_loss import mse_loss, compare

def test():
    x = torch.rand((3, 1, 4, 4), requires_grad=True)
    t = torch.ones((3, 1, 3, 3))

    net = nn.Conv2d(1, 1, 2)
    y = net(x)

    err = nn.MSELoss()(y, t)
    err.backward()
    de_dx = x.grad

    x, t = x.detach().numpy(), t.detach().numpy()

    net2 = Conv(1, 1, 2)
    net2.copy_torch(net)
    y2 = net2.forward(x)

    err2, de_dx2 = mse_loss(y2, t)
    de_dx2 = net2.backward(de_dx2, 0)

    compare('y', y, y2)
    compare('err', err, err2)
    compare('de_dx', de_dx, de_dx2)
    compare('de_dw', net.weight.grad, net2.de_dw)
    compare('de_db', net.bias.grad, net2.de_db)

if __name__=='__main__':
    test()
