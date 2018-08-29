import client

def init(dataset, filename, train_cut):
    global Client
    Client = client.Client(dataset, filename, train_cut)

# returns flattened gradient
def privateFun():
    return Client.getGrad()

def simpleStep(gradient):
    Client.simpleStep(gradient)


### Extra functions
def updateMOdel(modelWeights):
    Client.updateModel(modelWeights)

def getModel():
    return Client.getModel()

def getTestErr():
    return Client.getTestErr()