from mnist_dataset import MNISTDataset
from lfw_dataset import LFWDataset


def get_dataset(dataset):
    if dataset == "mnist":
        return MNISTDataset
    elif dataset == "lfw":
        return LFWDataset
    else: 
        print("Error: dataset " + dataset + "not defined")
    

def get_num_features(dataset):
    if dataset == "mnist":
        return 784
    elif dataset == "lfw":
        return 8742 #62 47 3
    else:
        print("Error: dataset " + dataset + "not defined")
    
def get_num_classes(dataset):
    if dataset == "mnist":
        return 10
    elif dataset == "lfw":
        return 12
    else: 
        print("Error: dataset " + dataset + "not defined")
        