from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pdb
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



#Our batch shape for input x is (62, 47, 3)
class CIFARDataset(Dataset):
    def __init__(self, filename, root_dir, transform=None, is_train=True, train_cut=.80):
        self.filename = filename
        self.root_dir = root_dir
        self.transform = transform
        data = np.load(os.path.join(root_dir, filename + ".npy"))
        n, d = data.shape

        cut = int(n*train_cut)
        if is_train:
            self.X = data[0:cut, 0:d - 1]
            self.y = data[0:cut, -1]
        else:
            self.X = data[cut:n, 0:d - 1]
            self.y = data[cut:n, -1]


    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        sample = np.reshape(self.X[idx], (32, 32, 3))

        if self.transform:
            sample = self.transform(sample)
        
        return {'image': sample, 'label': self.y[idx]}
    
    def getData(self):
        pdb.set_trace()
        return self.X, self.y

