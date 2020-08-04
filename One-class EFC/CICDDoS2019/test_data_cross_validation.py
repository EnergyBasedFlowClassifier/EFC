import sys
import numpy as np
from classification_functions import *
from matplotlib import pyplot as plt
from cutoff import define_cutoff
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

def plot_energies(file, sets, cutoff, energies):
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


    plt.savefig("Cross_validation/Discretized/{}/Sets{}/energies.png".format(file, sets), format="png")
    plt.close('all')


# this script performs the classification of the test set.
def test_data_function(file, sets):
    TEST_FILE = "Cross_validation/Discretized/{}/Sets{}/test.npy".format(file, sets)
    TRAIN_FILE = "Cross_validation/Discretized/{}/Sets{}/train.npy".format(file, sets)
    Q = 14
    LAMBDA = 0.5

    normal_data = list(np.load(TRAIN_FILE, allow_pickle=True))

    model_normal, h_i = create_model(np.array(normal_data,dtype=int), Q, LAMBDA)
    np.save("Cross_validation/Discretized/{}/Sets{}/h_i".format(file, sets), h_i)
    np.save("Cross_validation/Discretized/{}/Sets{}/model_normal".format(file, sets), model_normal)

    test_data = np.array(list(np.load(TEST_FILE, allow_pickle=True)))

    expected_results = np.array(list(np.load("Cross_validation/Discretized/{}/Sets{}/test_labels.npy".format(file, sets))))
    expected_results = [0 if x==1 else 1 for x in expected_results]

    CUTOFF = define_cutoff(TRAIN_FILE, file, sets)

    predicted_labels, energies = test_model(test_data, model_normal, h_i, expected_results, CUTOFF, Q)
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies).reshape(-1,1))]

    precision = precision_score(expected_results, predicted_labels)
    recall = recall_score(expected_results, predicted_labels)
    f1 = f1_score(expected_results, predicted_labels)
    roc = roc_auc_score(expected_results, predict_prob)

    np.save("Cross_validation/Discretized/{}/Sets{}/results.npy".format(file, sets), np.array([precision, recall, f1, roc]))
    np.save("Cross_validation/Discretized/{}/Sets{}/energies.npy".format(file, sets), np.array(energies))
    np.save("Cross_validation/Discretized/{}/Sets{}/cutoff.npy".format(file, sets), np.array(CUTOFF))