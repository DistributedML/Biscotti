from __future__ import division
from numpy.linalg import norm
import numpy as np
import utils
import pdb
import emcee

lammy = 0.01
verbose = 1
X = 0
y = 0
iteration = 1
alpha = 1e-2
d = 0
hist_grad = 0
epsilon = 0

scale = False
diffPriv13 = False
diffPriv16 = True
expected_iters = 100

def init(dataset, filename, epsilon, batch_size):

    passedEpsilon = epsilon
    data = utils.load_dataset(dataset)

    global X
    X = data['X']

    global y
    y = data['y']

    global d
    d = X.shape[1]

    global hist_grad
    hist_grad = np.zeros(d)

    global samples
    samples = []

    global this_batch_size
    this_batch_size = batch_size

    def lnprob(x, alpha):
        return -(alpha / 2) * np.linalg.norm(x)

    # nwalkers = max(4 * d, 250)
    # sampler = emcee.EnsembleSampler(nwalkers, d, lnprob, args=[passedEpsilon])

    # p0 = [np.random.rand(d) for i in range(nwalkers)]
    # pos, _, state = sampler.run_mcmc(p0, 100)

    # sampler.reset()
    # sampler.run_mcmc(pos, 1000, rstate0=state)

    # print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

    # samples = sampler.flatchain

    if epsilon > 0:

        if diffPriv13:

            nwalkers = max(4 * d, 250)
            sampler = emcee.EnsembleSampler(nwalkers, d, lnprob, args=[epsilon])

            p0 = [np.random.rand(d) for i in range(nwalkers)]
            pos, _, state = sampler.run_mcmc(p0, 100)

            sampler.reset()
            sampler.run_mcmc(pos, 1000, rstate0=state)

            print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))

            samples = sampler.flatchain

        elif diffPriv16:
            
            sigma = np.sqrt(2 * np.log(1.25)) / epsilon
            noise = sigma * np.random.randn(batch_size, expected_iters, d)
            samples = np.sum(noise, axis=0)

    else:

        samples = np.zeros((expected_iters, d))

    return d


def funObj(ww, X, y, batch_size):
    yXw = y * X.dot(ww)

    # Calculate the function value
    f = np.sum(np.logaddexp(0, -yXw)) + 0.5 * lammy * ww.T.dot(ww)

    # Calculate the gradient value
    res = - y / np.exp(np.logaddexp(0, yXw))
    if scale:
        g = (1 / batch_size) * X.T.dot(res) / \
            max(1, np.linalg.norm(X.T.dot(res))) + lammy * ww
    else:
        g = (1 / batch_size) * X.T.dot(res) + lammy * ww

    return f, g

def getNoise(iteration):
    return (-alpha / this_batch_size) * samples[iteration % len(samples)]

# Reports the direct change to w, based on the given one.
# Batch size could be 1 for SGD, or 0 for full gradient.
def privateFun(ww):

    global iteration
    ww = np.array(ww)

    # Define constants and params
    nn, dd = X.shape

    if this_batch_size > 0 and this_batch_size < nn:
        idx = np.random.choice(nn, this_batch_size, replace=False)
    else:
        # Just take the full range
        idx = range(nn)

    w_new = ww
    for i in range(5):
        f, g = funObj(ww, X[idx, :], y[idx], this_batch_size)

        d1, _ = samples.shape

        if diffPriv13 or diffPriv16:
            delta = -alpha * g + getNoise(iteration)
        else:
            delta = -alpha * g

            w_new = w_new + delta
            f_new, g_new = funObj(w_new, X[idx, :], y[idx], this_batch_size)
    iteration = iteration + 1

    return delta

if __name__ == '__main__':
    
    batch_size = 10
    epsilon = 1
    init("creditcard0", "creditcard0", epsilon, batch_size)
    ww = np.zeros(d)

    for i in range(3000):
    
        grad = privateFun(ww)
        delta = grad

        if (np.any(np.isnan(delta))):
            pdb.set_trace()

        ww = ww + delta

    pdb.set_trace()

