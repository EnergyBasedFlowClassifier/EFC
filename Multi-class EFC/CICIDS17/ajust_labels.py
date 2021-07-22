import pandas as pd
import numpy as np
import os
import sys

def ajust_sets(j):
    malicious_names = ['BENIGN',  'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack',
     'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']

    test_labels = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(j+1), header=None))
    for i, value in enumerate(malicious_names):
        test_labels[test_labels == value] = i
    test_labels = [int(x) for x in test_labels]
    np.savetxt("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(j+1), test_labels, delimiter=',')

    train_labels = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(j+1), header=None))
    for i, value in enumerate(malicious_names):
        train_labels[train_labels == value] = i
    train_labels = [int(x) for x in train_labels]
    np.savetxt("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(j+1), train_labels, delimiter=',')

    test_labels = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(j+1), header=None))
    test_labels = [int(x) for x in test_labels]
    np.savetxt("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(j+1), test_labels, delimiter=',')

    train_labels = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/train_labels.csv".format(j+1), header=None))
    train_labels = [int(x) for x in train_labels]
    np.savetxt("5-fold_sets/Discretized/Sets{}/train_labels.csv".format(j+1), train_labels, delimiter=',')

for j in range(5):
    ajust_sets(j)
