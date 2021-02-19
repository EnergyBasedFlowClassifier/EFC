import pandas as pd
import numpy as np
import os

malicious_names = ['normal','pingScan','bruteForce','portScan','dos']

for j in range(5):
    test = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test.csv".format(j+1)))
    np.save("5-fold_sets/Non_discretized/Sets{}/test.npy".format(j+1), test)

    test_labels = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(j+1)))
    for i, value in enumerate(malicious_names):
        test_labels[test_labels == value] = i
    test_labels = [int(x) for x in test_labels]
    print(np.unique(test_labels))
    np.save("5-fold_sets/Non_discretized/Sets{}/test_labels.npy".format(j+1), test_labels)


    train = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train.csv".format(j+1)))
    np.save("5-fold_sets/Non_discretized/Sets{}/train.npy".format(j+1), train)

    train_labels = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(j+1)))
    for i, value in enumerate(malicious_names):
        train_labels[train_labels == value] = i
    train_labels = [int(x) for x in train_labels]
    print(np.unique(train_labels))
    np.save("5-fold_sets/Non_discretized/Sets{}/train_labels.npy".format(j+1), train_labels)

    os.remove("5-fold_sets/Non_discretized/Sets{}/test.csv".format(j+1))
    os.remove("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(j+1))
    os.remove("5-fold_sets/Non_discretized/Sets{}/train.csv".format(j+1))
    os.remove("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(j+1))


    test = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/test.csv".format(j+1)))
    np.save("5-fold_sets/Discretized/Sets{}/test.npy".format(j+1), test)

    test_labels = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(j+1)))
    test_labels = [int(x) for x in test_labels]
    np.save("5-fold_sets/Discretized/Sets{}/test_labels.npy".format(j+1), test_labels)

    train = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/train.csv".format(j+1)))
    np.save("5-fold_sets/Discretized/Sets{}/train.npy".format(j+1), train)

    train_labels = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/train_labels.csv".format(j+1)))
    train_labels = [int(x) for x in train_labels]
    np.save("5-fold_sets/Discretized/Sets{}/train_labels.npy".format(j+1), train_labels)

    os.remove("5-fold_sets/Discretized/Sets{}/test.csv".format(j+1))
    os.remove("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(j+1))
    os.remove("5-fold_sets/Discretized/Sets{}/train.csv".format(j+1))
    os.remove("5-fold_sets/Discretized/Sets{}/train_labels.csv".format(j+1))
