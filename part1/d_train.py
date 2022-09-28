import numpy as np
from a_utils import timer
from a_loss import cross_entropy_loss
from b_act import Relu
from c_full import Full
from c_conv import Conv
from c_module import Series
from d_data import get_data

train_data, test_data = get_data()

if 0:
    net = Series([ 
        Full(28*28, 512), Relu(),
        Full(512, 512),   Relu(),
        Full(512, 10)
    ])
else:
    net = Series([ 
        Conv(1, 8, 3, 2), Relu(),
        Conv(8, 8, 3, 2), Relu(),
        Full(np.prod((8, 6, 6)), 10),
    ])

@timer
def test():
    right = 0; tot = 0
    for x, t in test_data:
        y = net.forward(x)
        right += sum(np.argmax(y, 1) == t)
        tot += len(y)
    print(f'accuracy {right}/{tot}')

@timer
def train():
    tot = 0
    for x, t in train_data:
        y = net.forward(x)
        err, de_dy = cross_entropy_loss(y, t)
        net.backward(de_dy, lr=0.1)
        tot += len(y)

test()
for e in range(10):
    train()
    test()
