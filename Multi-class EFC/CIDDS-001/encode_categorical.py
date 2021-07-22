import pandas as pd
import numpy as np
import os

malicious_names = ['normal','pingScan','bruteForce','portScan','dos']

def encode(chunk, proto, flags):
    for x, string in enumerate(proto):
        chunk.iloc[:, 1] = [x if value == string else value for value in chunk.iloc[:,1]]
    for x, string in enumerate(flags):
        chunk.iloc[:, 6] = [x if value == string else value for value in chunk.iloc[:,6]]
    return chunk

for fold in range(1,6):
    dict = np.load("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), allow_pickle=True)
    proto, flags = dict[1], dict[6]
    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train.csv".format(fold), chunksize=6000000, header=None)
    for chunk in reader:
        data = encode(chunk, proto, flags)
        data.to_csv("5-fold_sets/Non_discretized/Sets{}/encoded_train.csv".format(fold), mode='a', header=False, index=False)

    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test.csv".format(fold), chunksize=6000000, header=None)
    for chunk in reader:
        data = encode(chunk, proto, flags)
        data.to_csv("5-fold_sets/Non_discretized/Sets{}/encoded_test.csv".format(fold), mode='a', header=False, index=False)

    # train_labels =  pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(fold), header=None)
    # test_labels =  pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(fold), header=None)
    # for i, value in enumerate(malicious_names):
    #     train_labels.iloc[:,-1][train_labels.iloc[:,-1] == value] = i
    #     test_labels.iloc[:,-1][test_labels.iloc[:,-1] == value] = i
    # print(np.unique(train_labels))
    # print(np.unique(test_labels))
    # train_labels.to_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(fold), header=False, index=False)
    # test_labels.to_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(fold), header=False, index=False)
