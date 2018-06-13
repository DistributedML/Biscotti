from __future__ import division
import numpy as np
import utils
import matplotlib.pyplot as plt
import pdb


class SoftMaxModelTest:
    def __init__(self, X_test, y_test, X_train, y_train, numClasses, numFeatures):
        self.X_test = X_test
        self.y_test = y_test
        self.X_train = X_train
        self.y_train = y_train
        self.numClasses = numClasses
        self.numFeatures = numFeatures


    def train_error(self, ww):    
        # hardcoded for MNIST
        W = np.reshape(ww, (self.numClasses, self.numFeatures))
        yhat = np.argmax(np.dot(self.X_train, W.T), axis=1)
        error = np.mean(yhat != self.y_train)
        return error

    def test_error(self, ww):
        W = np.reshape(ww, (self.numClasses, self.numFeatures))
        yhat = np.argmax(np.dot(self.X_test, W.T), axis=1)
        error = np.mean(yhat != self.y_test)
        return error, yhat


