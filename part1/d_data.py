from typing import List, Tuple
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
from a_utils import timer

@timer
def get_data(
    dataset = datasets.FashionMNIST,
    batch = 50, 
    limit = 2000,
    root = r"C:\Users\65837\Downloads"
) -> Tuple[List[List[np.ndarray]]]:    
    train_loader = DataLoader(dataset(
        root, train=True, download=True, transform=ToTensor(),
    ), batch_size=batch, shuffle=True)
    test_loader = DataLoader(dataset(
        root, train=False, download=True, transform=ToTensor(),
    ), batch_size=batch, shuffle=True)
    def to_numpy(loader):
        res = []
        for x, t in loader:
            res.append([x.numpy(), t.numpy()])
            if len(res*batch) >= limit: break
        return res
    return to_numpy(train_loader), to_numpy(test_loader)

if __name__=='__main__':
    train_data, test_data = get_data()
    for i, (x, t) in enumerate(train_data):
        print(i, x.shape, t.shape)
