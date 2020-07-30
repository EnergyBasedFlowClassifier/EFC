import numpy as np
import pandas as pd
import random
import os

#this script creates train and test sets, for each file used in the experiments, to be used in the domain adaptation
#experiment

def build_train_test(file):
    data = np.load("Discretized_unique/{}.npy".format(file), allow_pickle=True)
    normals = [x for x in data if x[-1] == 1]
    atack = [x for x in data if x[-1] != 1]
    normal_samples = random.Random(20).sample(normals, 1250+1250)
    atack_samples = random.Random(20).sample(atack, 1250+1250)

    train_normal = [x[1:-1] for x in normal_samples[:1250]]
    train_normal_idx = [int(x[0]) for x in normal_samples[:1250]]
    np.save("External_test/Discretized/{}/Train_normal.npy".format(file), train_normal)

    train_malicious = [x[1:-1] for x in atack_samples[:1250]]
    train_malicious_idx = [int(x[0]) for x in atack_samples[:1250]]
    np.save("External_test/Discretized/{}/Train_malicious.npy".format(file), train_malicious)

    test = [x[1:-1] for x in normal_samples[1250::] + atack_samples[1250::]]
    np.save("External_test/Discretized/{}/Test.npy".format(file), test)

    test_idx = [int(x[0]) for x in normal_samples[1250::] + atack_samples[1250::]]
    test_labels = [x[-1] for x in normal_samples[1250::] + atack_samples[1250::]]
    test_labels = [1 if x == 1 else 0 for x in test_labels]
    np.save("External_test/Discretized/{}/Test_labels.npy".format(file), test_labels)


    data = np.array(pd.read_csv("Pre_processed/{}.csv".format(file)))
    train_normal = data[[x for x in train_normal_idx], 1:-1]
    np.save("External_test/Non_discretized/{}/Train_normal.npy".format(file), train_normal)
    train_normal_labels = data[[x for x in train_normal_idx], -1]
    train_normal_labels = [1 if x == 'BENIGN' else 0 for x in train_normal_labels]
    np.save("External_test/Non_discretized/{}/Train_normal_labels.npy".format(file), train_normal_labels)
    train_malicious = data[[x for x in train_malicious_idx], 1:-1]
    np.save("External_test/Non_discretized/{}/Train_malicious.npy".format(file), train_malicious)
    train_malicious_labels = data[[x for x in train_malicious_idx], -1]
    train_malicious_labels = [1 if x == 'BENIGN' else 0 for x in train_malicious_labels]
    np.save("External_test/Non_discretized/{}/Train_malicious_labels.npy".format(file), train_malicious_labels)
    test = data[[x for x in test_idx], 1:-1]
    np.save("External_test/Non_discretized/{}/Test.npy".format(file), test)
    test_labels = data[[x for x in test_idx], -1]
    test_labels = [1 if x == 'BENIGN' else 0 for x in test_labels]
    np.save("External_test/Non_discretized/{}/Test_labels.npy".format(file), test_labels)

files = ['TFTP_01-12','Syn_03-11','DrDoS_NTP_01-12']
for file in files:
    build_train_test(file)
