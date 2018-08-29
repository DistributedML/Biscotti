import pdb
import sys
sys.path.append('datasets')
sys.path.append('models')
import softmax_model
from client import Client


# Initialize Clients
# First Client is the aggregator
def main():
    clients = []
    print("Creating clients")
    for i in range(10):
        # TODO: Separate train data into non-iid batches
        clients.append(Client("lfw", "lfw_maleness_train"+str(i)))
    test_client = Client("lfw", "lfw_maleness_test", 0)
    
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