import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class MNISTCNNModel(nn.Module):
    def __init__(self):
        super(MNISTCNNModel, self).__init__()
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 5, 1, 4), # output space (16, 16, 16)
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(2))
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(16*16*16, 10)

    def forward(self, x):
        # x = x.view(x.shape[0], 28, 28)
        # x = x.unsqueeze(1)

        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        # out = self.drop_out(out)
        out = self.fc1(out)
        return out
    
    def reshape(self, flat_gradient):
        layers = []
        layers.append( torch.from_numpy(np.reshape(flat_gradient[0:400], (16, 1, 5, 5))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[400:400+16], (16, ))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[400+16: 400+16 + 10*4096], (10, 4096))).type(torch.FloatTensor))
        layers.append( torch.from_numpy( np.reshape(flat_gradient[400+16 + 10*4096: 400+16 + 10*4096 + 10], (10, ))).type(torch.FloatTensor))
        return layers