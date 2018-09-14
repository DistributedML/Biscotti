import pdb
import sys
sys.path.append('datasets')
sys.path.append('models')
import softmax_model
from client import Client
from softmax_model import SoftmaxModel
from mnist_cnn_model import MNISTCNNModel
from lfw_cnn_model import LFWCNNModel
from svm_model import SVMModel
import datasets
import math
import matplotlib.pylab as mp
import matplotlib.pyplot as plt

def returnModel(D_in, D_out):
    # model = SoftmaxModel(D_in, D_out)
    model = LFWCNNModel()
    return model

# Initialize Clients
# First Client is the aggregator
def main():
    iter_time = 1000
    clients = []
    average_loss = []
    test_accuracy_rate = []
    D_in = datasets.get_num_features("lfw")
    D_out = datasets.get_num_classes("lfw")
    batch_size = 4
    train_cut = 0.8

    print("Creating clients")
    for i in range(10):
        model = returnModel(D_in, D_out)    
        clients.append(Client("lfw", "lfw_maleness_train" + str(i), batch_size, model, train_cut))

    model = returnModel(D_in, D_out)
    test_client = Client("lfw", "lfw_maleness_test", batch_size, model, 0)

    
    print("Training for iterations")
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
            average_loss.append(loss / len(clients))
            test_accuracy_rate.append(accuracy_rate)

    # plot average loss and accuracy rate of the updating model
    x = range(1, int(math.floor(iter_time / 100)) + 1)
    fig, ax1 = plt.subplots()
    ax1.plot(x, average_loss, color = 'orangered',label = 'lfw_gender_average_loss')
    plt.legend(loc = 2)
    ax2 = ax1.twinx()
    ax2.plot(x, test_accuracy_rate, color='blue', label = 'lfw_gender_test_accuracy_rate')
    plt.legend(loc = 1)
    ax1.set_xlabel("iteration time / 100")
    ax1.set_ylabel("average_loss")
    ax2.set_ylabel("accuracy_rate")
    plt.title("lfw_gender_graph")
    plt.legend()
    mp.show()

    test_client.updateModel(modelWeights)
    test_err = test_client.getTestErr()
    print("Test error: " + str(test_err))


if __name__ == "__main__":
    main()