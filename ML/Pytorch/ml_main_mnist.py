import pdb
from client import Client
from softmax_model import SoftmaxModel
from mnist_cnn_model import MNISTCNNModel
from lfw_cnn_model import LFWCNNModel
from svm_model import SVMModel
import datasets
import math
import matplotlib.pylab as mp
import matplotlib.pyplot as plt
import numpy as np

# Just a simple sandbox for testing out python code, without using Go.
def debug_signal_handler(signal, frame):
    pdb.set_trace()


def returnModel(D_in, D_out):
    model = SoftmaxModel(D_in, D_out)
    # model = MNISTCNNModel()
    return model

# Initialize Clients
# First Client is the aggregator
def main():
    iter_time = 2000
    clients = []
    test_accuracy_rate = []
    average_loss = []
    D_in = datasets.get_num_features("mnist")
    D_out = datasets.get_num_classes("mnist")
    batch_size = 10
    train_cut = 0.8

    for i in range(6):
        model = returnModel(D_in, D_out)
        clients.append(Client("mnist", "mnist" + str(i), batch_size, model, train_cut))

    for i in range(4):
        model = returnModel(D_in, D_out)
        clients.append(Client("mnist", "mnist_bad_full", batch_size, model, train_cut))

    model = returnModel(D_in, D_out)
    test_client = Client("mnist", "mnist_test", batch_size, model, 0)

    rejections = np.zeros(10)

    for iter in range(iter_time):
        
        modelWeights = clients[0].getModelWeights()
        # Calculate and aggregaate gradients    
        for i in range(10):
            grad = clients[i].getGrad()
            # roni = test_client.roni(modelWeights, grad)
            # print "Client " + str(i) + " RONI is " + str(roni)
            # if roni > 0.02:
            #     rejections[i] += 1
            clients[0].updateGrad(grad)
        
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

            test_client.updateModel(modelWeights)
            test_err = test_client.getTestErr()
            attack_rate = test_client.get17AttackRate()

            print("Average loss is " + str(loss / len(clients)))
            print("Test error: " + str(test_err))
            print("Attack rate on 1s: " + str(attack_rate) + "\n")
            
            average_loss.append(loss / len(clients))
            test_accuracy_rate.append(attack_rate)

    # plot average loss and accuracy rate of the updating model
    x = range(1, int(math.floor(iter_time / 100)) + 1)
    fig, ax1 = plt.subplots()
    ax1.plot(x, average_loss, color = 'orangered',label = 'mnist_average_loss')
    plt.legend(loc = 2)
    ax2 = ax1.twinx()
    ax2.plot(x, test_accuracy_rate, color='blue', label = 'mnist_test_accuracy_rate')
    plt.legend(loc = 1)
    ax1.set_xlabel("iteration time / 100")
    ax1.set_ylabel("average_loss")
    ax2.set_ylabel("accuracy_rate")
    plt.title("mnist_graph")
    plt.legend()
    mp.show()

    test_client.updateModel(modelWeights)
    test_err = test_client.getTestErr()
    print("Test error: " + str(test_err))
    accuracy_rate = 1 - test_err
    print("Accuracy rate: " + str(accuracy_rate))

if __name__ == "__main__":
    main()