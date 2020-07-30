import pandas as pd
import numpy as np
import random
import os

#this script creates the stratified sets for cross validation, for each file used in the experiments.
#it separate the discretized sets to be used in EFC and the correspondent non discretized sets to be used in
#the other classifiers

def build_sets(file):
    data = np.load("Discretized_unique/{}.npy".format(file), allow_pickle=True)
    atacks = random.Random(6).sample([x for x in data if x[-1] != 1], 2500)
    atacks_index = [int(x[0]) for x in atacks]
    normals = random.Random(6).sample([x for x in data if x[-1] == 1], 2500)
    normals_index = [int(x[0]) for x in normals]

    for cv in range(10):
        train_normals = normals[cv*250:(cv+1)*250]
        test_normals = normals[:cv*250] + normals[(cv+1)*250::]
        train_malicious = atacks[cv*250:(cv+1)*250]
        test_malicious = atacks[:cv*250] + atacks[(cv+1)*250::]

        train_malicious = [x[1:-1] for x in train_malicious]
        train_normals = [x[1:-1] for x in train_normals]
        test = [x[1:-1] for x in test_normals + test_malicious]
        labels = [x[-1] for x in test_normals + test_malicious]
        np.save("Cross_validation/Discretized/{}/Sets{}/train_malicious.npy".format(file, cv+1), np.array(train_malicious))
        np.save("Cross_validation/Discretized/{}/Sets{}/train_normals.npy".format(file, cv+1), np.array(train_normals))
        np.save("Cross_validation/Discretized/{}/Sets{}/test.npy".format(file, cv+1), np.array(test))
        np.save("Cross_validation/Discretized/{}/Sets{}/test_labels.npy".format(file, cv+1), np.array(labels))


    data = np.array(pd.read_csv("Pre_processed/{}.csv".format(file)))

    atacks_pre = list(data[[x for x in atacks_index], :])
    normals_pre = list(data[[x for x in normals_index], :])

    for cv in range(10):
        train_normals = normals_pre[cv*250:(cv+1)*250]
        test_normals = normals_pre[:cv*250] + normals_pre[(cv+1)*250::]
        train_malicious = atacks_pre[cv*250:(cv+1)*250]
        test_malicious = atacks_pre[:cv*250] + atacks_pre[(cv+1)*250::]

        train = [x[1:-1] for x in train_normals] + [x[1:-1] for x in train_malicious]
        train_labels = [x[-1] for x in train_normals + train_malicious]
        test = [x[1:-1] for x in test_normals + test_malicious]
        test_labels = [x[-1] for x in test_normals + test_malicious]

        np.save("Cross_validation/Non-discretized/{}/Sets{}/train.npy".format(file, cv+1), np.array(train))
        np.save("Cross_validation/Non-discretized/{}/Sets{}/test.npy".format(file, cv+1), np.array(test))
        np.save("Cross_validation/Non-discretized/{}/Sets{}/test_labels.npy".format(file, cv+1), np.array(test_labels))
        np.save("Cross_validation/Non-discretized/{}/Sets{}/train_labels.npy".format(file, cv+1), np.array(train_labels))


files = ['TFTP_01-12','Syn_03-11','DrDoS_NTP_01-12']
for file in files:
    build_sets(file)
