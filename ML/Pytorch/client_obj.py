from __future__ import division
import numpy as np
import client
import pdb
import emcee
from softmax_model import SoftmaxModel
from mnist_cnn_model import MNISTCNNModel
from lfw_cnn_model import LFWCNNModel
from svm_model import SVMModel
import datasets

learning_rate = 1e-2

# Use the Song and Sarwate 2013 implementation. Will likely crash if d is too large
# Otherwise, use the 2016 Abadi.
diffPriv13 = False
expected_iters = 5000

def init(dataset, filename, epsilon, batch_size):

    
    global myclient

    D_in = datasets.get_num_features(dataset)
    D_out = datasets.get_num_classes(dataset)
    nParams = datasets.get_num_params(dataset)

    model = SoftmaxModel(D_in, D_out)
    train_cut = 0.8
    
    myclient = client.Client(dataset, filename, batch_size, learning_rate, model, train_cut)

    global samples
    samples = []

    global this_batch_size
    this_batch_size = batch_size

    def lnprob(x, alpha):
        return -(alpha / 2) * np.linalg.norm(x)

    if epsilon > 0:

        if diffPriv13:

            nwalkers = max(4 * nParams, 250)
            sampler = emcee.EnsembleSampler(nwalkers, nParams, lnprob, args=[epsilon])

            p0 = [np.random.rand(nParams) for i in range(nwalkers)]
            pos, _, state = sampler.run_mcmc(p0, 100)

            sampler.reset()
            sampler.run_mcmc(pos, 1000, rstate0=state)

            print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

            samples = sampler.flatchain

        else:

            sigma = np.sqrt(2 * np.log(1.25)) / epsilon
            noise = sigma * np.random.randn(batch_size, expected_iters, nParams)
            samples = np.sum(noise, axis=0)

    else:

        samples = np.zeros((expected_iters, nParams))

    return nParams

# returns flattened gradient
# Keep batch_size for API compliance
def privateFun(ww):
    global myclient
    weights = np.array(ww)
    myclient.updateModel(weights)
    return (-1) * myclient.getGrad()

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

def getNoise(iteration):
    return (-1) * (learning_rate / this_batch_size) * samples[iteration]

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
    
    epsilon = 0
    batch_size = 10
    dim = init("mnist", "mnist_train", epsilon, batch_size)
    ww = np.zeros(dim)
    numRejected = 0

    for i in range(3000):
    
        grad = privateFun(ww)
        delta = grad + getNoise(i)

        if (np.any(np.isnan(delta))):
            pdb.set_trace()

        if i % 50 == 0:
            print("Test err: " + str(getTestErr(ww)))

        ww = ww + delta

    print(getTestErr(ww))
    # print("Num rejected by RONI: " + str(numRejected))

    pdb.set_trace()
