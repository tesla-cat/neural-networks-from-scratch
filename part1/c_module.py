from typing import List
from c_full import Full
import numpy as np

class Series:
    def __init__(s, units: List[Full]):
        s.units = units
    
    def forward(s, x):
        for u in s.units: x = u.forward(x)
        return x 
    
    def backward(s, de_dy, lr):
        for u in reversed(s.units): 
            de_dy = u.backward(de_dy, lr)

class Parallel(Series):
    def forward(s, x):
        return np.concatenate([
            u.forward(x_) for u, x_ in zip(s.units, x)
        ])

    def backward(s, de_dy, lr):
        return np.concatenate([ 
            u.backward(d_, lr) for u, d_ in zip(s.units, de_dy)
        ])
