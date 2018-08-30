import numpy as np
import client
import pdb

def init(dataset, filename):
    global Client
    Client = client.Client(dataset, filename, 0.8)
    
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
