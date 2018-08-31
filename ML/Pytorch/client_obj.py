import numpy as np
import client
import pdb
from softmax_model import SoftmaxModel
from mnist_cnn_model import MNISTCNNModel
from lfw_cnn_model import LFWCNNModel
from svm_model import SVMModel
import datasets


def init(dataset, filename):
    global Client

    D_in = datasets.get_num_features(dataset)
    D_out = datasets.get_num_classes(dataset)
    
    batch_size = 4
    model = SoftmaxModel(D_in, D_out)
    train_cut = 0.8
    
    Client = client.Client(dataset, filename, batch_size, model, train_cut)
    
    if dataset == "mnist":
    	return 7850
    elif dataset == "lfw":
    	return 18254

    return -50

# returns flattened gradient
def privateFun(ww):
	weights = np.array(ww)
	Client.updateModel(weights)
	return Client.getGrad()

def simpleStep(gradient):
    Client.simpleStep(gradient)


### Extra functions
def updateModel(modelWeights):
    Client.updateModel(modelWeights)

def getModel():
    return Client.getModel()

def getTestErr(ww):
	weights = np.array(ww)
	Client.updateModel(weights)
	return Client.getTestErr()

def roni(ww, delta):
    
    weights = np.array(ww)
    update = np.array(delta)

    # Get original score
    Client.updateModel(weights)
    original = Client.getTestErr()

    Client.updateModel(weights + update)
    after = Client.getTestErr()

    return after - original

if __name__ == '__main__':
    
    dim = init("mnist", "mnist_train")
    ww = np.zeros(dim)

    grad = privateFun(ww)

    pdb.set_trace()
