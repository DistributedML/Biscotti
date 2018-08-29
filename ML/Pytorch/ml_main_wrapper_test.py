import sys
sys.path.append('datasets')
sys.path.append('models')
import client_obj
import pdb

client_obj.init("mnist", "mnist_train", train_cut=0.8)

pdb.set_trace()
grad = client_obj.privateFun()
client_obj.simpleStep(grad)