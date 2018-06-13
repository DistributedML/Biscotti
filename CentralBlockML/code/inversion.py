import pdb
import numpy as np

# grad: 1 x d np array
# if we know that values are between 
def invert(grad, numClasses, numFeatures):
    print("Inverting")
    # scale so max value is 2.55
    return np.reshape(grad[4], (numClasses, numFeatures))[1] * 2.55/np.amax(grad)