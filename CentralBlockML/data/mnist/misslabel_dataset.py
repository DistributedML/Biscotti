import numpy as np
import sys
import os

''' Mislabels preprocessed data
Takes input
    - dataset: mnist amazon kddcup
    - from class: eg 4
    - to class : eg 9
 and saves updated dataset as "bad_[dataset]_[from]_[to].npy"
'''

def main(argv):

    dataset = argv[0]
    fromClass = argv[1]
    toClass = argv[2]
    filename = dataset + fromClass + '.npy'


    data = np.load(os.path.join(filename))
    y = data[:, -1] 
    y[y==int(fromClass)] = int(toClass)
    save_file = dataset + "_bad_" + fromClass + "_" + toClass
    print("Generated : " + save_file)
    np.save(save_file, data)

if __name__ == "__main__":
    main(sys.argv[1:])
