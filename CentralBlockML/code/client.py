import softmax_model_test
import softmax_model_obj
import numpy as np
from scipy import spatial
import utils
import pdb
np.set_printoptions(suppress=True)


class Client:
    def __init__(self, dataset, epsilon, numClasses, numFeatures, batchSize):
        print("initializing client")
        data = utils.load_dataset(dataset, npy=True)
        X = data['X']
        y = data['y']
        (n, d) = X.shape
        X_train = X[range(0, n/2), :]
        y_train = y[range(0, n/2)]
        X_test = X[range(n/2, n), :]
        y_test = y[range(n/2, n)]
        self.ytest = y_test
        self.batchSize = batchSize
        self.softmax_model_obj = softmax_model_obj.SoftMaxModel(X_train, y_train, epsilon, numClasses)
        self.softmax_model_test = softmax_model_test.SoftMaxModelTest(X_test, y_test, X_train, y_train, numClasses, numFeatures)

    def submitGradient(self, models):
      
        bestSim = -1
        # Select the branch with the most similar generated gradient
        for branch in models:
            branchGrad = self.softmax_model_obj.privateFun(1, branch.getWeights(), self.batchSize)
            # use cosine similarity for similarity
            sim = spatial.distance.cosine(branchGrad, branch.getPreviousGrad())
            if sim > bestSim:
                bestSim = sim
                bestBranch = branch
                bestBranchGrad = branchGrad
        
        bestBranch.submitGradient(bestBranchGrad)

    def updateModels(self, models):
        print("fetching models")
        self.models = models

    def getBestBranch(self, models):
        bestErr = 1
        bestBranch = models[0]
        for branch in models:
            testErr, _ = self.softmax_model_test.test_error(branch.getWeights())
            # self.softmax_model_test.train_error(branch.getWeights())
            if testErr < bestErr:
                bestErr = testErr
                bestBranch = branch
        return bestBranch, bestErr

    def test(self, models):
        bestBranch, bestErr = self.getBestBranch(models)        
        print("Best test error for client is: " + str(bestErr))


    def poisoningCompare(self, models, fromClass, toClass):
        bestBranch, bestErr = self.getBestBranch(models)                
        error, yhat = self.softmax_model_test.test_error(bestBranch.getWeights())

        targetIdx = np.where(self.ytest == fromClass)
        if targetIdx[0].size == 0: 
            return
        attacked = np.mean(yhat[targetIdx] == toClass)
        print("Target Attack Rate (" + str(fromClass) + " to " + str(toClass) + "): " + str(attacked)  + "\n")
