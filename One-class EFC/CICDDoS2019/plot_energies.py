import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#this script creates the energie histograms shown in the paper
#the first function creates the histograms for the cross validation
#and the second creates the histograms for external tests

def plot_energies_CV(file, a, b):
    color_attack = "#006680"
    color_normal = "#b3b3b3"
    data_normal = np.array([])
    data_atk = np.array([])
    cutoff = 0
    for i in range(1,11):
        energies = np.load("Cross_validation/Discretized/{}/Sets{}/energies.npy".format(file, i), allow_pickle=True)
        data_normal = np.concatenate((data_normal, energies[:int(len(energies)/2)]), axis=0)
        data_atk = np.concatenate((data_atk, energies[int(len(energies)/2):]), axis=0)
        cutoff += np.load("Cross_validation/Discretized/{}/Sets{}/cutoff.npy".format(file, i))

    cutoff = cutoff/10
    weights = np.array([1]*len(data_normal))/len(data_normal)
    bins = np.histogram(np.hstack((data_normal,data_atk)), bins=30)[1]
    plt.xlim(a, b)
    plt.ylim(0,.6)
    plt.hist(data_atk, bins, weights=weights, facecolor=color_attack, alpha=0.7, label="malicious", ec='white', linewidth=0.3)
    plt.hist(data_normal, bins, weights=weights, facecolor=color_normal, alpha=0.7, label="benign", ec='white', linewidth=0.3)
    plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Probability", fontsize=12)

    plt.savefig("Cross_validation/Results/{}/efc_energies.png".format(file), format="png")
    plt.show()


def plot_energies_external(file, a, b):
    color_attack = "#006680"
    color_normal = "#b3b3b3"
    data_normal = np.array([])
    data_atk = np.array([])
    cutoff = 0
    for i in range(1,11):
        energies = np.load("External_test/Results/Training_{}/Sets{}/energies.npy".format(file, i), allow_pickle=True)
        data_normal = np.concatenate((data_normal, energies[:int(len(energies)/2)]), axis=0)
        data_atk = np.concatenate((data_atk, energies[int(len(energies)/2):]), axis=0)
        cutoff += np.load("Cross_validation/Discretized/{}/Sets{}/cutoff.npy".format(file, i))

    cutoff = cutoff/10
    weights = np.array([1]*len(data_normal))/len(data_normal)
    bins = np.histogram(np.hstack((data_normal,data_atk)), bins=30)[1]
    plt.xlim(a, b)
    plt.ylim(0,.6)
    plt.hist(data_atk, bins, weights=weights, facecolor=color_attack, alpha=0.7, label="malicious", ec='white', linewidth=0.3)
    plt.hist(data_normal, bins, weights=weights, facecolor=color_normal, alpha=0.7, label="benign", ec='white', linewidth=0.3)
    plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.savefig("External_test/Results/Training_{}/efc_energies.png".format(file), format="png")

    plt.show()



plot_energies_CV('DrDoS_NTP_01-12', 700, 1500)
plot_energies_external('DrDoS_NTP_01-12', 700, 1500)

plot_energies_CV('TFTP_01-12', 700, 1500)
plot_energies_external('TFTP_01-12', 700, 1500)

plot_energies_CV('Syn_03-11', 700, 1500)
plot_energies_external('Syn_03-11', 700, 1500)
