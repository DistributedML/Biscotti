from sklearn.datasets import fetch_lfw_people
import numpy as np
import pdb

lfw_people = fetch_lfw_people(color=True, min_faces_per_person=50)
train_cut = 0.5

def parse_noniid():
    # split into test and train
    test_data_init = False

    for i in range(len(lfw_people.target_names)):
        print("Saving lfw" + str(i))
        idx = np.where((lfw_people.target == i))[0]
        class_slice = lfw_people.data[idx]
        target_slice = lfw_people.target[idx]
        data_slice = np.hstack((class_slice, target_slice[:, None]))
        n, d = data_slice.shape
        pdb.set_trace()
        # np.save("lfw"+str(i), data_slice[0:int(n*train_cut)])

        if not test_data_init:
            test_data = data_slice[int(n*train_cut):]
            test_data_init = True
        else:
            test_data = np.vstack((test_data, data_slice[int(n*train_cut):]))
  
    np.save("lfw_test", test_data)

if __name__ == "__main__":
    parse_noniid()

