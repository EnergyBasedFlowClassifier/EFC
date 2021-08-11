import pandas as pd
import numpy as np
import os

malicious_names = ['normal','pingScan','bruteForce','portScan','dos']

def encode(chunk, dict):
    for feature in [1, 6]:
        diff = np.setdiff1d(chunk.iloc[:, feature], dict[feature])
        if diff.shape[0] > 0:
            dict[feature] += [x for x in diff]
        for x, string in enumerate(dict[feature]):
            chunk.iloc[:, feature] = [x if value == string else value for value in chunk.iloc[:,feature]]

    return chunk

for fold in range(1,6):
    dict = np.load("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), allow_pickle=True)
    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), chunksize=6000000, header=None)
    for chunk in reader:
        data = encode(chunk, dict)
        data.to_csv("5-fold_sets/Non_discretized/Sets{}/encoded_train.csv".format(fold), mode='a', header=False, index=False)

    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test.csv".format(fold), chunksize=6000000, header=None)
    for chunk in reader:
        data = encode(chunk, dict)
        data.to_csv("5-fold_sets/Non_discretized/Sets{}/encoded_test.csv".format(fold), mode='a', header=False, index=False)

    train_labels =  pd.read_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(fold), header=None)
    test_labels =  pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(fold), header=None)
    for i, value in enumerate(malicious_names):
        train_labels.iloc[:,-1][train_labels.iloc[:,-1] == value] = i
        test_labels.iloc[:,-1][test_labels.iloc[:,-1] == value] = i
    print(np.unique(train_labels))
    print(np.unique(test_labels))
    train_labels.to_csv("5-fold_sets/Non_discretized/Sets{}/encoded_train_labels.csv".format(fold), header=False, index=False)
    test_labels.to_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(fold), header=False, index=False)
