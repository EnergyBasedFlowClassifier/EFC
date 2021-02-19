import pandas as pd
import numpy as np
import random

for i in range(5):
    print(i)
    #discretized
    train = np.load("5-fold_sets/Discretized/Sets{}/train.npy".format(i+1), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Discretized/Sets{}/train_labels.npy".format(i+1), allow_pickle=True)

    normals_index = random.Random(1).sample(list(np.where(train_labels==0)[0]), 6000)
    pingscan_index = list(np.where(train_labels==1)[0])
    bruteforce_index = list(np.where(train_labels==2)[0])
    portscan_index = random.Random(1).sample(list(np.where(train_labels==3)[0]), 6000)
    dos_index = random.Random(1).sample(list(np.where(train_labels==4)[0]), 6000)

    train = train[normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index, :]
    train_labels = train_labels[normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index]

    np.save("5-fold_sets/Discretized/Sets{}/reduced_train.npy".format(i+1), train)
    np.save("5-fold_sets/Discretized/Sets{}/reduced_train_labels.npy".format(i+1), train_labels)


    #Non_discretized
    train = np.load("5-fold_sets/Non_discretized/Sets{}/train.npy".format(i+1), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Non_discretized/Sets{}/train_labels.npy".format(i+1), allow_pickle=True)

    train = train[normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index, :]
    train_labels = train_labels[normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index]

    np.save("5-fold_sets/Non_discretized/Sets{}/reduced_train.npy".format(i+1), train)
    np.save("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.npy".format(i+1), train_labels)
