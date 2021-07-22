import pandas as pd
import numpy as np
import random
import os

for fold in range(1,6):
    print(fold)
    #discretized
    train = pd.read_csv("5-fold_sets/Discretized/Sets{}/train.csv".format(fold), header=None)
    train_labels = pd.read_csv("5-fold_sets/Discretized/Sets{}/train_labels.csv".format(fold), header=None)

    normals_index = random.Random(1).sample(list(np.where(train_labels==0)[0]), 6000)
    pingscan_index = list(np.where(train_labels==1)[0])
    bruteforce_index = list(np.where(train_labels==2)[0])
    portscan_index = random.Random(1).sample(list(np.where(train_labels==3)[0]), 6000)
    dos_index = random.Random(1).sample(list(np.where(train_labels==4)[0]), 6000)

    train = train.iloc[normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index, :]
    train_labels = train_labels.iloc[normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index]

    print("after", train.shape, train_labels.shape)

    train.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(fold), header=False, index=False)
    train_labels.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train_labels.csv".format(fold), header=False, index=False)

    #Non_discretized
    train = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/encoded_train.csv".format(fold), header=None)
    train_labels = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(fold), header=None)
    print("before", train.shape, train_labels.shape)

    train = train.iloc[normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index, :]
    train_labels = train_labels.iloc[normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index]
    print("after", train.shape, train_labels.shape)

    train.to_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), header=False, index=False)
    train_labels.to_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(fold), header=False, index=False)
