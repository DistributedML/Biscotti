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

# Initialize Clients
# First Client is the aggregator
def returnModel(D_in, D_out):
    # model = SoftmaxModel(D_in, D_out)
    model = LFWCNNModel()
    return model
    
def main():
    clients = []
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
    for iter in range(1000):
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
    pdb.set_trace()

if __name__ == "__main__":
    main()