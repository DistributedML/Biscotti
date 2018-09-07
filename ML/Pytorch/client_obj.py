import numpy as np
import client
import pdb
from softmax_model import SoftmaxModel
from mnist_cnn_model import MNISTCNNModel
from lfw_cnn_model import LFWCNNModel
from svm_model import SVMModel
import datasets


def init(dataset, filename):
    global myclient

    D_in = datasets.get_num_features(dataset)
    D_out = datasets.get_num_classes(dataset)
    params = datasets.get_num_params(dataset)

    batch_size = 4
    model = SoftmaxModel(D_in, D_out)
    train_cut = 0.8
    
    myclient = client.Client(dataset, filename, batch_size, model, train_cut)

    return params

# returns flattened gradient
def privateFun(ww):
    global myclient
    weights = np.array(ww)
    myclient.updateModel(weights)
    return -1 * myclient.getGrad()

def simpleStep(gradient):
    global myclient
    myclient.simpleStep(gradient)

### Extra functions
def updateModel(modelWeights):
    global myclient
    myclient.updateModel(modelWeights)

def getModel():
    return myclient.getModel()

def getTestErr(ww):
    global myclient
    weights = np.array(ww)
    myclient.updateModel(weights)
    return myclient.getTestErr()

def roni(ww, delta):
    global myclient
    weights = np.array(ww)
    update = np.array(delta)

    # Get original score
    myclient.updateModel(weights)
    original = myclient.getTestErr()

    myclient.updateModel(weights + update)
    after = myclient.getTestErr()

    return after - original

if __name__ == '__main__':
    
    dim = init("creditcard", "creditcardtrain")
    ww = np.zeros(dim)

    for i in range(1000):
        #print(ww)
        if i % 10 == 0:
            print("Test err: " + str(getTestErr(ww)))

        grad = privateFun(ww)
        ww = ww + grad

    print(getTestErr(ww))

    pdb.set_trace()
