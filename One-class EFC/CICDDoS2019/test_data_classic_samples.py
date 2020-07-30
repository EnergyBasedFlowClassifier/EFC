import sys
import numpy as np
from classification_functions import *
from matplotlib import pyplot as plt
from cutoff import define_cutoff
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler


def plot_energies(train_file, sets, cutoff, energies):
    color_attack = "#006680"
    color_normal = "#b3b3b3"
    data_normal = energies[:int(len(energies)/2)]
    data_atk = energies[int(len(energies)/2):]

    weights = np.array([1]*len(data_normal))/len(data_normal)
    bins = np.histogram(np.hstack((data_normal,data_atk)), bins=100)[1]
    plt.hist(data_normal, bins, weights=weights, facecolor=color_normal, alpha=0.7, label="normal")
    plt.hist(data_atk, bins, weights=weights, facecolor=color_attack, alpha=0.7, label="suspicious")
    plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Probability", fontsize=12)


    plt.savefig("External_test/Results/Training_{}/Sets{}/energies.png".format(train_file, sets), format="png")
    plt.close('all')


# this script performs the classification of the test set.
def test_data_function(train_file, sets, test_file1, test_file2):
    test1 = np.load("External_test/Discretized/{}/Test.npy".format(test_file1), allow_pickle=True)
    test2 = np.load("External_test/Discretized/{}/Test.npy".format(test_file2), allow_pickle=True)
    test_normals = np.concatenate((test1[:int(len(test1)/2),:], test2[:int(len(test2)/2),:]), axis=0)
    test_malicious = np.concatenate((test1[int(len(test1)/2):,:], test2[int(len(test2)/2):,:]), axis=0)
    test_data = np.concatenate((test_normals, test_malicious), axis=0)
    test_labels1 = np.load("External_test/Discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True)
    test_labels2 = np.load("External_test/Discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)
    test_labels_normals = np.concatenate((test_labels1[:int(len(test_labels1)/2)], test_labels2[:int(len(test_labels2)/2)]), axis=0)
    test_labels_malicious = np.concatenate((test_labels1[int(len(test_labels1)/2):], test_labels2[int(len(test_labels2)/2):]), axis=0)
    test_labels = np.concatenate((test_labels_normals, test_labels_malicious), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]
    Q = 14
    LAMBDA = 0.5


    model_normal = np.load("Cross_validation/Discretized/{}/Sets{}/model_normal.npy".format(train_file, sets), allow_pickle=True)
    h_i = np.load("Cross_validation/Discretized/{}/Sets{}/h_i.npy".format(train_file, sets), allow_pickle=True)
    CUTOFF = np.load("Cross_validation/Discretized/{}/Sets{}/cutoff.npy".format(train_file, sets), allow_pickle=True)

    predicted_labels, energies = test_model(test_data, model_normal, h_i, test_labels, CUTOFF, Q)
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies).reshape(-1,1))]

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    roc = roc_auc_score(test_labels, predict_prob)

    np.save("External_test/Results/Training_{}/Sets{}/results.npy".format(train_file, sets), np.array([precision, recall, f1, roc]))
    np.save("External_test/Results/Training_{}/Sets{}/energies.npy".format(train_file, sets), np.array(energies))
