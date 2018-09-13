import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pdb

class MNISTCNNModel(nn.Module):
    def __init__(self):
        super(MNISTCNNModel, self).__init__()

        # ONE LAYER
        self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 16, 5, 1, 4), # output space (16, 16, 16)
                                        torch.nn.ReLU(),
                                        torch.nn.MaxPool2d(1)
                                        )

        # TWO LAYERS
        # self.layer1 = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 5, 1, 4), # output space (8, 16, 16)
        #                                 torch.nn.ReLU(),
        #                                 torch.nn.MaxPool2d(2),

        #                                 torch.nn.Conv2d(8, 16, 5, 1, 4), # output space (16, 10, 10)
        #                                 torch.nn.ReLU(),
        #                                 torch.nn.MaxPool2d(2)
        # )

        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(16*32*32, 10) # ONE LAYER
        # self.fc1 = nn.Linear(16*10*10, 10) # TWO LAYERS

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
        # ONE LAYER
        l1 = 16*1*5*5
        l2 = 16
        l3 = 10*16*32*32
        l4 = 10

        layers.append( torch.from_numpy( np.reshape(flat_gradient[0:l1], (16, 1, 5, 5))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1:l1+l2], (l2, ))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2: l1+l2+l3], (10, 16*32*32))).type(torch.FloatTensor) )
        layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3 : l1+l2+l3+l4], (l4, ))).type(torch.FloatTensor) )

        # TWO LAYERS
        # l1 = 8*1*5*5
        # l2 = 8
        # l3 = 16*8*5*5
        # l4 = 16
        # l5 = 10*1600
        # l6 = 10

        # layers.append( torch.from_numpy( np.reshape(flat_gradient[0:l1], (8, 1, 5, 5))).type(torch.FloatTensor) )
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1:l1+l2], (l2, ))).type(torch.FloatTensor) )
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2: l1+l2+l3], (16, 8, 5, 5))).type(torch.FloatTensor) )
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3 : l1+l2+l3+l4], (l4, ))).type(torch.FloatTensor) )
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4: l1+l2+l3+l4+l5], (10, 1600))).type(torch.FloatTensor) )
        # layers.append( torch.from_numpy( np.reshape(flat_gradient[l1+l2+l3+l4+l5: l1+l2+l3+l4+l5+l6], (l6, ))).type(torch.FloatTensor) )
        return layers