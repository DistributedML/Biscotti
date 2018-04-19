from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt 

import logistic_model
import logistic_model_test
import numpy as np
import utils
import inversion_compare

import pdb


if __name__ == "__main__":

    dataset = "creditbad"
    outputName = "victimBlank.csv"

    batch_size = 10
    iterations = 4000
    epsilon = 5

    # Global
    numFeatures = logistic_model.init(dataset, epsilon=epsilon)
    
    for i in range(5):
        weights = np.random.rand(numFeatures)

        train_progress = np.zeros(iterations)
        test_progress = np.zeros(iterations)

        for i in xrange(iterations):
            deltas = logistic_model.privateFun(1, weights, batch_size)
            weights = weights + deltas
            #train_progress[i] = logistic_model_test.train_error(weights)
            #test_progress[i] = logistic_model_test.test_error(weights)

        # plt.plot(train_progress, "green")
        # plt.plot(test_progress, "red")

        #plt.show()

        np.savetxt(outputName, weights, delimiter=',')
        print("Train error: %d", logistic_model_test.train_error(weights))
        print("Test error: %d", logistic_model_test.test_error(weights))

        inversion_compare.compare(outputName)
