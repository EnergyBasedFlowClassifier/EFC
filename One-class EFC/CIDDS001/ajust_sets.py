import numpy as np
import pandas as pd
import shutil
import os

for i in range(1,11):
    # # #ajusta teste externo nao discretizado
    test = np.array(pd.read_csv("test_sets_cidds001/non-discretized/{}_testing_cidds_randomall_wext.csv".format(i)))
    test_labels = test[:,0]
    test_labels = [1 if x=='suspicious' else 0 for x in test_labels]

    test = np.delete(test, 0, axis=1)
    np.save("External_test/Non_discretized/Exp{}/external_test.npy".format(i), np.array(test))
    np.save("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(i), np.array(test_labels))
    #os.remove("test_sets_cidds001/non-discretized/{}_testing_cidds_randomall_wext.csv".format(i))

    test = np.load("test_sets_cidds001/discretized/{}_test_randomall_wext.npy".format(i), allow_pickle=True)
    test_labels = [0 for x in test[:len(test)//2]] + [1 for x in test[len(test)//2::]]
    np.save("External_test/Discretized/Exp{}/external_test.npy".format(i), np.array(test))
    np.save("External_test/Discretized/Exp{}/external_test_labels.npy".format(i), np.array(test_labels))
    #os.remove("test_sets_cidds001/discretized/{}_test_randomall_wext.npy".format(i))

    set = np.array(pd.read_csv("test_sets_cidds001/non-discretized/{}_testing_cidds_randomall_wall.csv".format(i)))
    set_labels = list(set[:,0])
    set_labels = [0 if x=='normal' else 1 for x in set_labels]
    print(np.unique(test_labels, return_counts=True))
    set = list(np.delete(set, 0, axis=1))
    train = set[:8000] + set[10000:18000]
    train_labels = set_labels[:8000] + set_labels[10000:18000]
    test = set[8000:10000] + set[18000::]
    test_labels = set_labels[8000:10000] + set_labels[18000::]
    np.save("Data/Non_discretized/Exp{}/train.npy".format(i), np.array(train))
    np.save("Data/Non_discretized/Exp{}/train_labels.npy".format(i), np.array(train_labels))
    np.save("Data/Non_discretized/Exp{}/test.npy".format(i), np.array(test))
    np.save("Data/Non_discretized/Exp{}/test_labels.npy".format(i), np.array(test_labels))
    #os.remove("test_sets_cidds001/non-discretized/{}_testing_cidds_randomall_wall.csv".format(i))

    set = list(np.load("test_sets_cidds001/discretized/{}_test_randomall_wall.npy".format(i), allow_pickle=True))
    set_labels = [0 for x in set[:len(set)//2]] + [1 for x in set[len(set)//2::]]

    train = set[:8000] + set[10000:18000]
    train_labels = set_labels[:8000] + set_labels[10000:18000]
    test = set[8000:10000] + set[18000::]
    test_labels = set_labels[8000:10000] + set_labels[18000::]

    np.save("Data/Discretized/Exp{}/train.npy".format(i), np.array(train))
    np.save("Data/Discretized/Exp{}/train_labels.npy".format(i), np.array(train_labels))
    np.save("Data/Discretized/Exp{}/test.npy".format(i), np.array(test))
    np.save("Data/Discretized/Exp{}/test_labels.npy".format(i), np.array(test_labels))
    #os.remove("test_sets_cidds001/discretized/{}_test_randomall_wall.npy".format(i))
