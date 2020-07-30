import pandas as pd
import numpy as np
import random
import os

#this script creates the stratified sets for cross validation, for each file used in the experiments.
#it separate the discretized sets to be used in EFC and the correspondent non discretized sets to be used in
#the other classifiers

def build_cv_friday():
    data = np.load("Discretized_unique/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.npy", allow_pickle=True)
    data_pre = np.array(pd.read_csv("Pre_processed/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"))

    atacks = random.Random(6).sample([x for x in data if x[-1] == 2], 5000)
    atacks_index = [x[0] for x in atacks]
    normals = random.Random(6).sample([x for x in data if x[-1] == 1], 5000)
    normals_index = [x[0] for x in normals]

    atacks_pre = list(data_pre[[x for x in atacks_index], :])
    normals_pre = list(data_pre[[x for x in normals_index], :])

    for cv in range(10):
        test_normals = normals[cv*500:(cv+1)*500]
        train_normals = normals[:cv*500] + normals[(cv+1)*500::]
        test_malicious = atacks[cv*500:(cv+1)*500]
        train_malicious = atacks[:cv*500] + atacks[(cv+1)*500::]

        train_malicious = [x[1:-1] for x in train_malicious]
        train_normals = [x[1:-1] for x in train_normals]
        test = [x[1:-1] for x in test_normals + test_malicious]
        test_labels = [x[-1] for x in test_normals + test_malicious]

        np.save("Cross_validation/Discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Sets{}/train_malicious.npy".format(cv+1), np.array(train_malicious))
        np.save("Cross_validation/Discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Sets{}/train.npy".format(cv+1), np.array(train_normals))
        np.save("Cross_validation/Discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Sets{}/test.npy".format(cv+1), np.array(test))
        np.save("Cross_validation/Discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Sets{}/test_labels.npy".format(cv+1), np.array(test_labels))

        test_normals = normals_pre[cv*500:(cv+1)*500]
        train_normals = normals_pre[:cv*500] + normals_pre[(cv+1)*500::]
        test_malicious = atacks_pre[cv*500:(cv+1)*500]
        train_malicious = atacks_pre[:cv*500] + atacks_pre[(cv+1)*500::]

        train = [x[1:-1] for x in train_normals] + [x[1:-1] for x in train_malicious]
        train_labels = [x[-1] for x in train_normals + train_malicious]
        test = [x[1:-1] for x in test_normals + test_malicious]
        test_labels = [x[-1] for x in test_normals + test_malicious]

        np.save("Cross_validation/Non-discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Sets{}/train.npy".format(cv+1), np.array(train))
        np.save("Cross_validation/Non-discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Sets{}/test.npy".format(cv+1), np.array(test))
        np.save("Cross_validation/Non-discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Sets{}/test_labels.npy".format(cv+1), np.array(test_labels))
        np.save("Cross_validation/Non-discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Sets{}/train_labels.npy".format(cv+1), np.array(train_labels))

def build_cv_wednesday():
    data = np.load("Discretized_unique/Wednesday-workingHours.pcap_ISCX.npy", allow_pickle=True)
    data_pre = np.array(pd.read_csv("Pre_processed/Wednesday-workingHours.pcap_ISCX.csv"))

    mixed2_atacks = random.Random(6).sample([x for x in data if x[-1] == 11], 2100)
    mixed2_index = [x[0] for x in mixed2_atacks]
    mixed2_atacks2 = random.Random(6).sample([x for x in data if x[-1] == 12], 2100)
    mixed2_index2 = [x[0] for x in mixed2_atacks2]
    mixed2_atacks3 = random.Random(6).sample([x for x in data if x[-1] == 13], 370)
    mixed2_index3 = [x[0] for x in mixed2_atacks3]
    mixed2_atacks4 = random.Random(6).sample([x for x in data if x[-1] == 14], 430)
    mixed2_index4 = [x[0] for x in mixed2_atacks4]
    normals = random.Random(6).sample([x for x in data if x[-1] == 1], 5000)
    normals_index = [x[0] for x in normals]

    mixed2_atacks_pre = list(data_pre[[x for x in mixed2_index], :])
    mixed2_atacks2_pre = list(data_pre[[x for x in mixed2_index2], :])
    mixed2_atacks3_pre = list(data_pre[[x for x in mixed2_index3], :])
    mixed2_atacks4_pre = list(data_pre[[x for x in mixed2_index4], :])
    normals_pre = list(data_pre[[x for x in normals_index], :])

    for cv in range(10):
        test_normals = normals[cv*500:(cv+1)*500]
        train_normals = normals[:cv*500] + normals[(cv+1)*500::]
        test_malicious = mixed2_atacks[cv*210:(cv+1)*210] + mixed2_atacks2[cv*210:(cv+1)*210]
        test_malicious += mixed2_atacks3[cv*37:(cv+1)*37] + mixed2_atacks4[cv*43:(cv+1)*43]
        train_malicious = mixed2_atacks[:cv*210] + mixed2_atacks[(cv+1)*210::] + mixed2_atacks2[:cv*210] + mixed2_atacks2[(cv+1)*210::]
        train_malicious += mixed2_atacks3[:cv*37] + mixed2_atacks3[(cv+1)*37::] + mixed2_atacks4[:cv*43] + mixed2_atacks4[(cv+1)*43::]

        train_malicious = [x[1:-1] for x in train_malicious]
        train_normals = [x[1:-1] for x in train_normals]
        test = [x[1:-1] for x in test_normals + test_malicious]
        test_labels = [x[-1] for x in test_normals + test_malicious]

        np.save("Cross_validation/Discretized/Wednesday-workingHours.pcap_ISCX/Sets{}/train_malicious.npy".format(cv+1), np.array(train_malicious))
        np.save("Cross_validation/Discretized/Wednesday-workingHours.pcap_ISCX/Sets{}/train.npy".format(cv+1), np.array(train_normals))
        np.save("Cross_validation/Discretized/Wednesday-workingHours.pcap_ISCX/Sets{}/test.npy".format(cv+1), np.array(test))
        np.save("Cross_validation/Discretized/Wednesday-workingHours.pcap_ISCX/Sets{}/test_labels.npy".format(cv+1), np.array(test_labels))

        test_normals = normals_pre[cv*500:(cv+1)*500]
        train_normals = normals_pre[:cv*500] + normals_pre[(cv+1)*500::]
        test_malicious = mixed2_atacks_pre[cv*210:(cv+1)*210] + mixed2_atacks2_pre[cv*210:(cv+1)*210]
        test_malicious += mixed2_atacks3_pre[cv*37:(cv+1)*37] + mixed2_atacks4_pre[cv*43:(cv+1)*43]
        train_malicious = mixed2_atacks_pre[:cv*210] + mixed2_atacks_pre[(cv+1)*210::] + mixed2_atacks2_pre[:cv*210] + mixed2_atacks2_pre[(cv+1)*210::]
        train_malicious += mixed2_atacks3_pre[:cv*37] + mixed2_atacks3_pre[(cv+1)*37::] + mixed2_atacks4_pre[:cv*43] + mixed2_atacks4_pre[(cv+1)*43::]

        train = [x[1:-1] for x in train_normals] + [x[1:-1] for x in train_malicious]
        train_labels = [x[-1] for x in train_normals + train_malicious]
        test = [x[1:-1] for x in test_normals + test_malicious]
        test_labels = [x[-1] for x in test_normals + test_malicious]

        np.save("Cross_validation/Non-discretized/Wednesday-workingHours.pcap_ISCX/Sets{}/train.npy".format(cv+1), np.array(train))
        np.save("Cross_validation/Non-discretized/Wednesday-workingHours.pcap_ISCX/Sets{}/test.npy".format(cv+1), np.array(test))
        np.save("Cross_validation/Non-discretized/Wednesday-workingHours.pcap_ISCX/Sets{}/test_labels.npy".format(cv+1), np.array(test_labels))
        np.save("Cross_validation/Non-discretized/Wednesday-workingHours.pcap_ISCX/Sets{}/train_labels.npy".format(cv+1), np.array(train_labels))

build_cv_friday()
build_cv_wednesday()
