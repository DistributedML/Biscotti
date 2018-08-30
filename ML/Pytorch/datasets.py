from mnist_dataset import MNISTDataset
from lfw_dataset import LFWDataset
from cifar_dataset import CIFARDataset

def get_dataset(dataset):
    if dataset == "mnist":
        return MNISTDataset
    elif dataset == "lfw":
        return LFWDataset
    elif dataset == "cifar":
        return CIFARDataset
    else: 
        print("Error: dataset " + dataset + "not defined")
    

def get_num_features(dataset):
    if dataset == "mnist":
        return 784
    elif dataset == "lfw":
        return 8742 #62 47 3
    elif dataset == "cifar":
        return 32*32*3
    else:
        print("Error: dataset " + dataset + "not defined")
    
def get_num_classes(dataset):
    if dataset == "mnist":
        return 10
    elif dataset == "lfw":
        return 12
    elif dataset == "cifar":
        return 10
    else: 
        print("Error: dataset " + dataset + "not defined")
        