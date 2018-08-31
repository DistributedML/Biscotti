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
class CreditDataset(Dataset):
    def __init__(self, filename, root_dir, transform=None, is_train=True, train_cut=.80):
        self.filename = filename
        self.root_dir = root_dir
        self.transform = transform
        
        df = pd.read_csv(os.path.join(root_dir, filename + ".csv"))
        nn, dd = df.shape

        # Need to remove the first column and first row
        credit = df.ix[1:nn, 1:dd].as_matrix()
        nn, dd = credit.shape

        datay = credit[:, dd - 1].astype(int)
        datay[np.where(datay == -1)] = 0
        idx = np.arange(nn)
        data = credit[idx, :].astype(float)

        cut = int(nn*train_cut)
        if is_train:
            self.X = data[:cut, :]
            self.y = datay[:cut]

        else:
            self.X = data[cut:, :]
            self.y = datay[cut:]
    
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        # sample = np.reshape(self.X[idx], (32, 32, 3))
        sample = self.X[idx]
        # if self.transform:
        #     sample = self.transform(sample)
        
        return {'image': sample, 'label': self.y[idx]}
    
    def getData(self):
        pdb.set_trace()
        return self.X, self.y

