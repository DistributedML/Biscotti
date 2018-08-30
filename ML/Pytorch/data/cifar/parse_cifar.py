import os
import torch
import numpy as np
import torch.nn
import cPickle
import torch.nn.functional as F
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import pdb
def debug_signal_handler(signal, frame):
    pdb.set_trace()

import signal
signal.signal(signal.SIGINT, debug_signal_handler)


def download_CIFAR10():
    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True)
    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=True)

def load_batch_file(file_name):
    with open(file_name, 'rb') as f:
        dict = cPickle.load(f)
    X = dict['data']
    Y = dict['labels']
    X = X.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
    Y = np.array(Y)
    return X, Y


def load_CIFAR10_train(root):
    xs = [] # list
    ys = []
    # use list to append, then use array to numpy.concatenate
    for i in range(1, 6):
        f = os.path.join(root, 'data_batch_%d' %(i, ))
        X, Y = load_batch_file(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs) # axis = 0, combine them into a 50000 rows array
    Ytr = np.concatenate(ys) # axis = 0
    return Xtr, Ytr


def load_CIFAR10_test(root):
    Xts, Yts = load_batch_file(os.path.join(root, 'test_batch'))
    return Xts, Yts

def slice_for_tc():

    # download_CIFAR10()
    root = './cifar-10-batches-py/'
    images, labels = load_CIFAR10_train(root)
    images_test, labels_test = load_CIFAR10_test(root)

    n = len(images)
    d = len(images[0].flatten())
    t = len(images_test)

    pdb.set_trace()

    Xtrain = np.zeros((n, d))
    Xtest = np.zeros((t, d))

    ytrain = np.asarray(labels)
    ytest = np.asarray(labels_test)

    for i in range(n):
        Xtrain[i, :] = np.asarray(images[i].flatten())

    for q in range(t):
        Xtest[q, :] = np.asarray(images_test[q].flatten())

    # standardize each column
    print("Standardize columns")
    Xtrain = Xtrain / 100.0
    # Xtrain, _, _ = standardize_cols(Xtrain)
    # Xtest, _, _ = standardize_cols(Xtest)

    for k in range(10):

        idx = np.where((ytrain == k))[0]

        class_slice = Xtrain[idx]
        data_slice = np.hstack((class_slice, ytrain[idx][:, None]))

        print("slice " + str(k) + " is shape " + str(data_slice.shape))

        np.save("cifar" + str(k), data_slice)

    train_slice = np.hstack((Xtrain, np.reshape(ytrain, (len(ytrain), 1))))
    np.save("cifar_train", train_slice)

    test_slice = np.hstack((Xtest, np.reshape(ytest, (len(ytest), 1))))
    np.save("cifar_test", test_slice)

    pdb.set_trace()


def standardize_cols(X, mu=None, sigma=None):
    # Standardize each column with mean 0 and variance 1
    n_rows, n_cols = X.shape

    if mu is None:
        mu = np.mean(X, axis=0)

    if sigma is None:
        sigma = np.std(X, axis=0)
        sigma[sigma < 1e-8] = 1.

    return (X - mu) / sigma, mu, sigma


if __name__ == "__main__":
    slice_for_tc()