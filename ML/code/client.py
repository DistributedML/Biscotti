from softmax_model import SoftmaxModel
from mnist_cnn_model import MNISTCNNModel
from lfw_cnn_model import LFWCNNModel
from svm_model import SVMModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score
import numpy as np
import pdb
import datasets

class Client():
    def __init__(self, dataset, filename, train_cut=.80):
        # initializes dataset
        self.batch_size=4
        Dataset = datasets.get_dataset(dataset)
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.trainset = Dataset(filename, "data/" + dataset, is_train=True, train_cut=train_cut, transform=transform)
        self.testset = Dataset(filename, "data/" + dataset, is_train=False, train_cut=train_cut, transform=transform)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=len(self.testset), shuffle=False)

        D_in = datasets.get_num_features(dataset)
        D_out = datasets.get_num_classes(dataset)

        self.model = SoftmaxModel(D_in, D_out)
        #self.model = MNISTCNNModel()
        # self.model = LFWCNNModel()

        # self.model = SVMModel(D_in, D_out)
        # self.criterion = nn.MultiLabelMarginLoss()
        ### Tunables ###
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.001)
        self.aggregatedGradients = []
        self.loss = 0.0

    # TODO:: Get noise for diff priv
    def getGrad(self):
        for i, data in enumerate(self.trainloader, 0):
            # get the inputs
            inputs = data['image'].float()
            labels = data['label'].long()

            # for svm
            # padded_labels = torch.zeros(self.batch_size,10).long()
            # padded_labels.transpose(0,1)[labels] = 1
            # labels = padded_labels 

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            nn.utils.clip_grad_norm(self.model.parameters(), 100)
            self.loss = loss.item()

            # TODO: Find more efficient way to flatten params
            # get gradients into layers
            layers = np.zeros(0)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    layers = np.concatenate((layers, param.grad.numpy().flatten()), axis=None)
            return layers

    
    # Called when an aggregator receives a new gradient
    def updateGrad(self, gradient):
        # Reshape into original tensor
        layers = self.model.reshape(gradient)
        self.aggregatedGradients.append(layers)

    # Step in the direction of provided gradient. 
    # Used in BlockML when gradient is aggregated in Go
    def simpleStep(self, gradient):
        print("Simple step")
        layers = self.model.reshape(gradient)
        # Manually updates parameter gradients
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = layers[layer]
                layer += 1

        # Step in direction of parameter gradients
        self.optimizer.step()


    # Called when sufficient gradients are aggregated to generate updated model
    def step(self):
        # Aggregate gradients together in place        
        for i in range(1, len(self.aggregatedGradients)):
            gradients = self.aggregatedGradients[i]
            for g, gradient in enumerate(gradients):
                self.aggregatedGradients[0][g] += gradient

        # Average gradients
        for g, gradient in enumerate(self.aggregatedGradients[0]):
            gradient /= len(self.aggregatedGradients)

        # Manually updates parameter gradients
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.aggregatedGradients[0][layer]
                layer += 1

        # Step in direction of parameter gradients
        self.optimizer.step()
        self.aggregatedGradients = []

    # Called when the aggregator shares the updated model
    def updateModel(self, modelWeights):
        layer = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = modelWeights[layer]
                layer += 1

    def getModelWeights(self):
        layers = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                layers.append(param.data) 
        return layers

    def getLoss(self):
        return self.loss
    
    def getModel(self):
        return self.model
    
    def getTestErr(self):
        for i, data in enumerate(self.testloader, 0):
            # get the inputs
            inputs = data['image'].float()
            labels = data['label'].long()
            inputs, labels = Variable(inputs), Variable(labels)
            out = self.model(inputs)
            pred = np.argmax(out.detach().numpy(), axis=1)
            return 1 - accuracy_score(pred, labels)

        # X, y = self.testset.getData()
        # X, y = Variable(torch.from_numpy(X)), Variable(torch.from_numpy(y))
        # pred = self.model(X.float())
        # y_hat = torch.argmax(pred, dim=1)
        # return 1 - accuracy_score(y, y_hat)

