import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class CIFARCNNModel(nn.Module):
    def __init__(self):
        super(CIFARCNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=5),   # output space (7, 10, 10) 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

            # nn.Conv2d(20, 40, kernel_size=3, stride=1, padding=5),  
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        )


        self.fc1 = nn.Linear(8000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return out
    
    
    # TODO: automatically reshape given model and data shape
    # For now, print out the grad.shape in client.getGrad
    def reshape(self, flat_gradient):
        layers = []
        # l1 = 20*3*3*3
        # l2 = 20
        # l3 = 40*20*3*3
        # l4 = 40
        # l5 = 10*7840
        # l6 = 10

        l1 = 20*3*3*3
        l2 = 20
        l3 = 10*8000
        l4 = 10
        layers.append( torch.from_numpy(np.reshape(flat_gradient[0:l1], (20, 3, 3, 3))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1:l1+l2], (l2, ))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2: l1+l2+l3], (10,8000))).type(torch.FloatTensor))
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3: l1+l2+l3+l4], (l4, ))).type(torch.FloatTensor))
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4: l1+l2+l3+l4+l5], (10, 7840))).type(torch.FloatTensor))
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4+l5: l1+l2+l3+l4+l5+l6], (l6, ))).type(torch.FloatTensor))
        return layers