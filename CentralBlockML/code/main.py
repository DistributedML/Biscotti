import pdb
import signal
import numpy as np
import inversion
import client
import modelBranch
from matplotlib import pyplot as plt

# setup debug interrupt
def debug_signal_handler(signal, frame):    
    pdb.set_trace()

signal.signal(signal.SIGINT, debug_signal_handler)

# Constants
ITER = 500
numClasses = 10
numFeatures = 784
numParams = numClasses * numFeatures
epsilon = 5
batchSize = 100

# Array of client
clients = []
# Array of modelBranch
models = []

# Initialize root model
initialW = np.random.rand(numParams) / 100.0
initialGrad = initialW
models.append(modelBranch.ModelBranch(initialW, initialGrad))

# Create healthy clients for each class
for i in range(0,10):
    clients.append(client.Client('mnist/mnist_unif'+str(i), epsilon, numClasses, numFeatures, batchSize))

# Create poisoners
clients.append(client.Client('mnist/mnist_unif_bad_4_9', epsilon, numClasses, numFeatures, batchSize))
clients.append(client.Client('mnist/mnist_unif_bad_4_9', epsilon, numClasses, numFeatures, batchSize))
clients.append(client.Client('mnist/mnist_unif_bad_4_9', epsilon, numClasses, numFeatures, batchSize))
clients.append(client.Client('mnist/mnist_unif_bad_4_9', epsilon, numClasses, numFeatures, batchSize))
clients.append(client.Client('mnist/mnist_unif_bad_4_9', epsilon, numClasses, numFeatures, batchSize))
clients.append(client.Client('mnist/mnist_unif_bad_4_9', epsilon, numClasses, numFeatures, batchSize))

# Train for ITER iterations
for i in xrange(ITER):
    print("iteration: " + str(i))
    # All clients submit gradients
    for client in clients:
        client.submitGradient(models)

    # All models branches update
    for branch in models:
        branch.updateModel()


# Check test accuracy
for client in clients:
    client.test(models)
    client.poisoningCompare(models, 4, 9)


### Try and link gradients together
invertedData = inversion.invert(models[0].gradientHistory[488], numClasses, numFeatures)
imgData = np.reshape(invertedData, (28, 28))
plt.imshow(imgData, cmap='gray')
plt.show()



