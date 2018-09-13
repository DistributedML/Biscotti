import pdb
from client import Client
from softmax_model import SoftmaxModel
from mnist_cnn_model import MNISTCNNModel
from lfw_cnn_model import LFWCNNModel
from svm_model import SVMModel
import datasets
import math
import matplotlib.pyplot as plt

def returnModel(D_in, D_out):
    # model = SoftmaxModel(D_in, D_out)
    model = MNISTCNNModel()
    return model

# Initialize Clients
# First Client is the aggregator
def main():
    iter_time = 2000
    clients = []
    test_accuracy_rate = []
    D_in = datasets.get_num_features("mnist")
    D_out = datasets.get_num_classes("mnist")
    batch_size = 10
    train_cut = 0.8

    for i in range(10):
        model = returnModel(D_in, D_out)
        clients.append(Client("mnist", "mnist" + str(i), batch_size, 0.01, model, train_cut))

    model = returnModel(D_in, D_out)
    test_client = Client("mnist", "mnist_test", batch_size, 0.01, model, 0)

    for iter in range(iter_time):
        # Calculate and aggregaate gradients    
        for i in range(10):
            clients[0].updateGrad(clients[i].getGrad())
        
        # Share updated model
        clients[0].step()
        modelWeights = clients[0].getModelWeights()
        for i in range(10):
            clients[i].updateModel(modelWeights)
        
        # Print average loss across clients
        if iter % 100 == 0:
            
            loss = 0.0
            for i in range(10):
                loss += clients[i].getLoss()

            print("Average loss is " + str(loss / len(clients)))
            test_client.updateModel(modelWeights)
            test_err = test_client.getTestErr()
            print("Test error: " + str(test_err))
            accuracy_rate = 1 - test_err
            print("Accuracy rate: " + str(accuracy_rate) + "\n")
            test_accuracy_rate.append(accuracy_rate)

    # plot accuracy rate of the updating model
    x = range(1, int(math.floor(iter_time / 100)) + 1)
    plt.plot(x, test_accuracy_rate, label = 'mnist-test-accuracy-rate')
    plt.xlabel("iteration time / 100")
    plt.ylabel("accuracy")
    plt.title("mnist-test-accuracy-graph")
    plt.legend()
    plt.show()

    test_client.updateModel(modelWeights)
    test_err = test_client.getTestErr()
    print("Test error: " + str(test_err))
    accuracy_rate = 1 - test_err
    print("Accuracy rate: " + str(accuracy_rate))

if __name__ == "__main__":
    main()