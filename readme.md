
# neural networks from scratch

## part 1: time independent neurons + backpropagation 

### full connection (same as `torch.nn.Linear`)

- [verification by comparing with pytorch](part1/c_full.py)

```python
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
```

### convolutional (sparse) connection (same as `torch.nn.Conv2d`)

- [verification by comparing with pytorch](part1/c_conv.py)

```python
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
```

### training using FashionMNIST

- [part1/d_train.py](part1/d_train.py)

```python
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

```

- output

```cmd
                     get_data took 0.250 s
accuracy 171/2000
                     test took 1.192 s
                     train took 2.823 s
accuracy 1071/2000
                     test took 0.346 s
                     train took 1.564 s
accuracy 1274/2000
                     test took 0.347 s
                     train took 1.580 s
accuracy 1334/2000
                     test took 0.342 s
                     train took 1.549 s
accuracy 1360/2000
                     test took 0.339 s
                     train took 1.571 s
accuracy 1396/2000
                     test took 0.351 s
                     train took 1.546 s
accuracy 1411/2000
                     test took 0.342 s
                     train took 1.548 s
accuracy 1442/2000
                     test took 0.347 s
                     train took 1.573 s
accuracy 1464/2000
                     test took 0.340 s
                     train took 1.549 s
accuracy 1480/2000
                     test took 0.342 s
                     train took 1.563 s
accuracy 1490/2000
                     test took 0.342 s
```