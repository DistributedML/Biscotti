import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class LFWCNNModel(nn.Module):
    def __init__(self):
        super(LFWCNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 18, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(18, 36, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )


        self.fc1 = nn.Linear(5940, 2)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return out
    
    # TODO: automatically reshape given model and data shape
    # For now, print out the grad.shape in client.getGrad
    def reshape(self, flat_gradient):
        layers = []
        l1 = 18*3*3*3
        l2 = 18
        l3 = 36*18*3*3
        l4 = 36
        l5 = 2*5940
        l6 = 2

        print l1 + l2 + l3 + l4 + l5 + l6

        layers.append( torch.from_numpy(np.reshape(flat_gradient[0:l1], (18, 3, 3, 3))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1:l1+l2], (18, ))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2: l1+l2+l3], (36,18,3,3))).type(torch.FloatTensor))
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3: l1+l2+l3+l4], (36, ))).type(torch.FloatTensor))
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4: l1+l2+l3+l4+l5], (2, 5940))).type(torch.FloatTensor))
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4+l5: l1+l2+l3+l4+l5+l6], (2, ))).type(torch.FloatTensor))
        return layers