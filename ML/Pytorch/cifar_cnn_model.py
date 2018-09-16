import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb


class CIFARCNNModel(nn.Module):
    def __init__(self):
        super(CIFARCNNModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=3),    
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1, stride=1, padding=0),

            # nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=3),  
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
        )

        self.fc1 = nn.Linear(25920, 10) # ONE LAYER
        # self.fc1 = nn.Linear(25600, 10) # TWO LAYERS

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        return out
    
    
    # TODO: automatically reshape given model and data shape
    # For now, print out the grad.shape in client.getGrad
    def reshape(self, flat_gradient):
        layers = []
        # ONE LAYER
        l1 = 20*3*3*3
        l2 = 20
        l3 = 10*25920
        l4 = 10

        layers.append( torch.from_numpy(np.reshape(flat_gradient[0:l1], (20, 3, 3, 3))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1:l1+l2], (l2, ))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2: l1+l2+l3], (10,25920))).type(torch.FloatTensor))
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3: l1+l2+l3+l4], (l4, ))).type(torch.FloatTensor))

        # TWO LAYERS
        # l1 = 8*3*3*3
        # l2 = 8
        # l3 = 16*8*3*3
        # l4 = 16
        # l5 = 10*25600
        # l6 = 10

        # layers.append( torch.from_numpy(np.reshape(flat_gradient[0:l1], (8, 3, 3, 3))).type(torch.FloatTensor) )
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1:l1+l2], (l2, ))).type(torch.FloatTensor) )
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2: l1+l2+l3], (16, 8, 3, 3))).type(torch.FloatTensor))
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3: l1+l2+l3+l4], (l4, ))).type(torch.FloatTensor))
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4: l1+l2+l3+l4+l5], (10, 25600))).type(torch.FloatTensor))
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4+l5: l1+l2+l3+l4+l5+l6], (l6, ))).type(torch.FloatTensor))

        return layers