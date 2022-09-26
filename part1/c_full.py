import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn

def full_forward(x, w, b):
    N = x.shape[0]
    x = np.reshape(x, (N, 1, -1))
    w = np.expand_dims(w, 0)
    b = np.expand_dims(b, 0)
    y = np.sum(w * x, axis=2) + b
    return y

def full_backward(x, w, de_dy):
    N = x.shape[0]
    x2 = np.reshape(x, (N, 1, -1))
    de_dy2 = np.expand_dims(de_dy, 2)
    w = np.expand_dims(w, 0)
    de_dx = np.sum(de_dy2 * w, axis=1).reshape(x.shape)
    de_dw = np.sum(de_dy2 * x2, axis=0) 
    de_db = np.sum(de_dy, axis=0) 
    return de_dx, de_dw, de_db

class Full:
    def __init__(s, H1, H2, w=None, b=None):
        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear
        k = np.sqrt(1/H1)
        s.w = npr.uniform(-k, k, (H2, H1)) if w is None else w
        s.b = npr.uniform(-k, k, (H2, ))   if b is None else b
    
    def copy_torch(s, x: nn.Linear):
        s.w = x.weight.detach().numpy()
        s.b = x.bias.detach().numpy()

    def forward(s, x):
        s.x = x
        return full_forward(x, s.w, s.b)
    
    def backward(s, de_dy, lr):
        de_dx, s.de_dw, s.de_db = full_backward(s.x, s.w, de_dy)
        s.w -= lr * s.de_dw
        s.b -= lr * s.de_db
        return de_dx

#=======================================


from a_loss import mse_loss, compare

def test():
    x = torch.rand((3, 3), requires_grad=True)
    t = torch.ones((3, 4))

    net = nn.Linear(3, 4)
    y = net(x)

    err = nn.MSELoss()(y, t)
    err.backward()
    de_dx = x.grad

    x, t = x.detach().numpy(), t.detach().numpy()

    net2 = Full(3, 4)
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
