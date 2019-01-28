from __future__ import division
import numpy as np
import utils
import pdb

data = utils.load_dataset("credittest")
Xvalid, yvalid = data['X'], data['y']

def kappa(ww, delta):
    
    ww = np.array(ww)
    yhat = np.sign(np.dot(Xvalid, ww))
    
    ww2 = np.array(ww + delta)
    yhat2 = np.sign(np.dot(Xvalid, ww2))

    P_A = np.sum(yhat == yhat2) / float(yvalid.size)
    P_E = 0.5

    return (P_A - P_E) / (1 - P_E)

def roni(ww, delta):
    
    ww = np.array(ww)
    yhat = np.sign(np.dot(Xvalid, ww))
    
    ww2 = np.array(ww + delta)
    yhat2 = np.sign(np.dot(Xvalid, ww2))

    g_err = np.sum(yhat != yvalid) / float(yvalid.size)
    new_err = np.sum(yhat2 != yvalid) / float(yvalid.size)

    return new_err - g_err

# Returns the index of the row that should be used in Krum
def krum(deltas, clip):

    # assume deltas is an array of size group * d
    n = len(deltas)

    scores = get_krum_scores(deltas, n - clip)

    good_idx = np.argpartition(scores, n - clip)[:(n - clip)]

    print(good_idx)

    return good_idx

    # return np.mean(deltas[good_idx], axis=0)


def get_krum_scores(X, groupsize):

    krum_scores = np.zeros(len(X))

    # Calculate distances
    distances = np.sum(X**2, axis=1)[:, None] + np.sum(
        X**2, axis=1)[None] - 2 * np.dot(X, X.T)

    for i in range(len(X)):
        krum_scores[i] = np.sum(np.sort(distances[i])[1:(groupsize - 1)])

    return krum_scores


if __name__ == "__main__":
	pdb.set_trace()