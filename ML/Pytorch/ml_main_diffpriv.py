import pdb
import sys
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
import numpy as np
from PIL import Image
import os
cwd = os.getcwd()
os.chdir("../../DistSys")   

delta = 0.00001

def returnModel(D_in, D_out):
    model = SoftmaxModel(D_in, D_out)
    # model = LFWCNNModel()
    return model

# rescales a numpy array to be in [a, b]
def rescale(x, a, b):
    minNum = np.min(x)
    maxNum = np.max(x)
    return (b - a)*(x - minNum) / (maxNum - minNum) + a 

def showImage(grad, dataset, dataclass, batch_size, epsilon):

    if dataset == "mnist":
        reshaped = np.reshape(grad[dataclass*784:(dataclass+1)*784], (28,28))
    elif dataset == "cifar":
        dataclass = 5
        reshaped = np.reshape( rescale(grad[32*32*3*dataclass:32*32*3*(dataclass+1)], 0, 2.5), (32,32, 3))

        reshaped = np.transpose(np.reshape(grad[0:32*32*3], (3,32,32)), (1,2,0))
    elif dataset == "lfw":
        reshaped = np.reshape( grad[0:8742], (62, 47, 3))

    
    if (dataset == 'mnist'):
        plt.imshow(reshaped, cmap='gray')
        img = reshaped
    elif (dataset == 'cifar'):
        plt.imshow(reshaped)
    
    else:
        from skimage import color
        from skimage import io
        img = color.rgb2gray(reshaped)

        def rgb2gray(rgb):
            return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

        plt.imshow(img, cmap=plt.get_cmap('gray'))
    os.chdir(cwd)   
    plt.axis('off')
    plt.savefig(dataset + "_batch_" + str(batch_size) + "_epsilon_" + str(epsilon) + ".png", bbox_inches='tight', pad_inches = 0)
    plt.show()
    os.chdir("../../DistSys") 
    return img

def varyEpsilonShowImage(grad, batch_size, epsilon, dataset, data_class):    
  
    # sigma = np.sqrt(2 * np.log(1.25)) / epsilon
    sigma = np.sqrt(2 * (np.log(1.25/delta))) / epsilon
    noise = sigma * np.random.randn(batch_size, nParams)
    samples = np.sum(noise, axis=0)
    showImage(grad + samples , dataset, data_class, batch_size, epsilon)
    
# Initialize Clients
# First Client is the aggregator
def main(batchsize,epsilon,dataset,data_filename,dataclass):
    
    train_cut = 0.8

    iter_time = 1
    clients = []
    average_loss = []
    test_accuracy_rate = []
    D_in = datasets.get_num_features(dataset)
    D_out = datasets.get_num_classes(dataset)

    global batch_size
    batch_size = batchsize  
    
    global nParams 
    nParams = datasets.get_num_params(dataset)    

    print("Creating clients")
    # for i in range(10):
    #     model = returnModel(D_in, D_out)    
    #     clients.append(Client("lfw", "lfw_maleness_train" + str(i), batch_size, model, train_cut))
    model = returnModel(D_in, D_out)

    # clients.append(Client("lfw", "lfw_maleness_person61_over20", batch_size, model, train_cut))
    clients.append(Client(dataset, data_filename, batch_size, model, train_cut))
    
    print("Training for iterations")
    
    for iter in range(iter_time):
        # Calculate and aggregate gradients    
        for i in range(1):
            grad= clients[i].getGrad()
            if epsilon== 0:
                showImage(grad, dataset,dataclass, batch_size, 0)
            else:
                varyEpsilonShowImage(grad,batch_size, epsilon, dataset,data_class)
            # showImage(grad, dataset,dataclass, batch_size, 0)
            # clients[0].updateGrad(noisegrad)       

if __name__ == "__main__":

    dataset = "mnist"
    data_filename = "mnist_digit5"
    data_class=5
    batch_sizes=["1", "100", "200", "300", "350"]
    # epsilons=["0.01", "0.1" , "0.5", "1", "2", "0"]
    epsilons=["0"]


    # dataset = "cifar"
    # dataclass = "cifar5"

    for batchsize in batch_sizes:
        for epsilon in epsilons:
            print("Batchsize: " + str(batchsize))
            main(int(batchsize),float(epsilon),dataset,data_filename,data_class)



####Training code

    ### Parameters ###
    # dataset = "mnist"
    # dataclass = "mnist_digit0"
    
    # dataset = "cifar"
    # dataclass = "cifar5"
    # dataclass = "lfw_maleness_person61_over20"

        # # Share updated model
        # clients[0].step()
        # modelWeights = clients[0].getModelWeights()
        # for i in range(1):
        #     clients[i].updateModel(modelWeights)
        
        # # Print average loss across clients
        # if iter % 100 == 0:
        #     loss = 0.0
        #     for i in range(1):
        #         loss += clients[i].getLoss()
        #     print("Average loss is " + str(loss / len(clients)))
        #     test_client.updateModel(modelWeights)
        #     test_err = test_client.getTestErr()
        #     print("Test error: " + str(test_err))
        #     accuracy_rate = 1 - test_err
        #     print("Accuracy rate: " + str(accuracy_rate) + "\n")
        #     average_loss.append(loss / len(clients))
        #     test_accuracy_rate.append(accuracy_rate)

    # # plot average loss and accuracy rate of the updating model
    # x = range(1, int(math.floor(iter_time / 100)) + 1)
    # fig, ax1 = plt.subplots()
    # ax1.plot(x, average_loss, color = 'orangered',label = 'lfw_gender_average_loss')
    # plt.legend(loc = 2)
    # ax2 = ax1.twinx()
    # ax2.plot(x, test_accuracy_rate, color='blue', label = 'lfw_gender_test_accuracy_rate')
    # plt.legend(loc = 1)
    # ax1.set_xlabel("iteration time / 100")
    # ax1.set_ylabel("average_loss")
    # ax2.set_ylabel("accuracy_rate")
    # plt.title("lfw_gender_graph")
    # plt.legend()
    # mp.show()

    # test_client.updateModel(modelWeights)
    # test_err = test_client.getTestErr()
    # print("Test error: " + str(test_err))
    # accuracy_rate = 1 - test_err
    # print("Accuracy rate: " + str(accuracy_rate) + "\n")

    # def varyEpsilonShowImage(grad, x, y, step):
    # pdb.set_trace()
  
    # # for epsilon in range(x, y, step):
    # sigma = np.sqrt(2 * np.log(1.25)) / epsilon
    # noise = sigma * np.random.randn(batch_size, nParams)
    # samples = np.sum(noise, axis=0)

    # showImage(grad + samples, epsilon)
    #     # plt.imshow(image, cmap=plt.get_cmap('gray'))
    #     # plt.show()
