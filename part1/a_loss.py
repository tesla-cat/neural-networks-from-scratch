import numpy as np

def mse_loss(x, t):
    err = np.mean((x-t)**2)
    de_dx = 1/np.prod(x.shape) * 2 * (x-t)
    return err, de_dx

#======================================

def one_hot(x, n, low=0, high=1): # tested
    y = np.ones((len(x), n)) * low
    y[range(len(x)), x] = high
    return y

#================================

def softmax(x): # tested
    y = np.exp(x)
    return y / np.sum(y, axis=1)[:, None]
    
def cross_entropy_loss(x, t):
    # https://www.michaelpiseno.com/blog/2021/softmax-gradient
    sm = softmax(x)
    mask = range(len(t))
    err = 1/len(x) * np.sum(- np.log( sm[mask, t] ))
    sm[mask, t] -= 1
    de_dx = 1/len(x) * sm
    return err, de_dx

#=======================================

import torch
import torch.nn as nn 

def compare(name, a: torch.Tensor, b: np.ndarray):
    d = np.abs(a.detach().numpy() - b)
    print(f'{name} \t diff_min {d.min():.6f} \t diff_max {d.max():.6f}')

def test_loss(nn_func, my_func, t):
    print('\n'+ my_func.__name__)
    x = torch.rand((3, 3), requires_grad=True)
    #t = torch.ones_like(x)

    err = nn_func(x, t)
    err.backward()
    de_dx = x.grad

    x, t = x.detach().numpy(), t.detach().numpy()
    err2, de_dx2 = my_func(x, t)
    compare('err', err, err2)
    compare('de_dx', de_dx, de_dx2)

if __name__=='__main__':
    test_loss(nn.MSELoss(), mse_loss, torch.ones((3,3)))
    test_loss(nn.CrossEntropyLoss(), cross_entropy_loss, torch.tensor([0,1,2]))
