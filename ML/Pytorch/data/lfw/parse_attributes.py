import pdb
import numpy as np

# parses lfw_attributes file and matches people by their attributes
people = {}

attributes = np.loadtxt('lfw_attributes.txt', dtype=np.str, delimiter='\t')

n,d = attributes.shape
for i in range(n):
    person = attributes[i]
    name = person[0]
    maleness = float(person[2])
    if people.has_key(name):
        people[name].append(maleness)
    else:
        people[name] = [maleness]

# need to call .item after load
np.save('lfw_maleness.npy', people)
