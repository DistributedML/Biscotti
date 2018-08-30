import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class SoftmaxModel(nn.Module):
    def __init__(self, D_in, D_out):
        super(SoftmaxModel, self).__init__()
        self.linear = nn.Linear(D_in, D_out)
        self.D_in = D_in
        self.D_out = D_out

    def forward(self, x):
        # TODO: fix hardcoded
        x = np.reshape(x, (x.shape[0], self.D_in))
        return self.linear(x)
    
    # Unflattens flattened gradient
    def reshape(self, flat_gradient):
        layers = []
        layers.append( torch.from_numpy(np.reshape(flat_gradient[0:self.D_in*self.D_out], (self.D_out, self.D_in))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy(flat_gradient[self.D_in*self.D_out:self.D_in*self.D_out + self.D_out]).type(torch.FloatTensor) )
        return layers
