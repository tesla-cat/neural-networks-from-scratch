from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from a_utils import timer
from a_loss import cross_entropy_loss
from b_act import Relu
from c_full import Full
from c_conv import Conv
from c_module import Series

batch = 50
root = r"C:\Users\65837\Downloads"

train_data = datasets.FashionMNIST(
    root, train=True, download=True, transform=ToTensor(),
)
train_loader = DataLoader(train_data, batch_size=batch, shuffle=True)

test_data = datasets.FashionMNIST(
    root, train=False, download=True, transform=ToTensor(),
)
test_loader = DataLoader(test_data, batch_size=batch, shuffle=True)

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
    right = 0
    tot = 0
    for x, t in test_loader:
        x, t = x.numpy(), t.numpy()
        y = net.forward(x)
        right += sum(np.argmax(y, 1) == t)
        tot += len(y)
        if tot >= 2000: break
    print(f'accuracy {right}/{tot}')

@timer
def train():
    tot = 0
    for x, t in train_loader:
        x, t = x.numpy(), t.numpy()
        y = net.forward(x)
        err, de_dy = cross_entropy_loss(y, t)
        net.backward(de_dy, lr=0.1)
        tot += len(y)
        if tot >= 2000: break

test()
for e in range(10):
    train()
    test()
