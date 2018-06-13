import numpy as np
import pdb

class ModelBranch:
    def __init__(self, initialW, initialGrad):
        print("initializing model")
        self.chain = [[initialW, initialGrad]]
        self.pendingGradients = []
        self.gradientHistory = []
    
    def updateModel(self):
        ### TODO:: Refactor out ###
        acc = np.zeros(self.chain[0][0].size)
        numPending = len(self.pendingGradients)
        for grad in self.pendingGradients:
            acc += grad
        newGrad = acc / numPending
        ###
        newW = self.chain[-1][0] + newGrad
        self.chain.append([newW, newGrad]) 
        ### Testing to see if gradients can be linked ###
        self.gradientHistory.append(self.pendingGradients[:])
        ###       
        self.pendingGradients = []
    

    
    def getWeights(self):
        return self.chain[-1][0]
    
    def getPreviousGrad(self):
        return self.chain[-1][1]

    def submitGradient(self, grad):
        self.pendingGradients.append(grad)