import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def plot_energies_CV():
    color_attack = "#006680"
    color_normal = "#b3b3b3"
    data_normal = np.array([])
    data_atk = np.array([])
    cutoff = 0
    for i in range(1,11):
        energies = np.load("Cross_validation/Discretized/Sets{}/energies.npy".format(i), allow_pickle=True)
        data_normal = np.concatenate((data_normal, energies[:int(len(energies)/2)]), axis=0)
        data_atk = np.concatenate((data_atk, energies[int(len(energies)/2):]), axis=0)
        cutoff += np.load("Cross_validation/Discretized/Sets{}/cutoff.npy".format(i))

    cutoff = cutoff/10
    weights = np.array([1]*len(data_normal))/len(data_normal)
    bins = np.histogram(np.hstack((data_normal,data_atk)), bins=30)[1]
    plt.xlim(0,350)
    plt.ylim(0,.6)
    plt.hist(data_atk, bins, weights=weights, facecolor=color_attack, alpha=0.7, label="malicious",  ec='white', linewidth=0.3)
    plt.hist(data_normal, bins, weights=weights, facecolor=color_normal, alpha=0.7, label="normal",  ec='white', linewidth=0.3)
    plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Probability", fontsize=12)


    plt.savefig("Cross_validation/Results/efc_energies.png", format="png")
    plt.show()


def plot_energies_external():
    color_attack = "#006680"
    color_normal = "#b3b3b3"
    data_normal = np.array([])
    data_atk = np.array([])
    cutoff = 0
    for i in range(1,11):
        energies = np.load("External_test/Results/Sets{}/energies.npy".format(i), allow_pickle=True)
        data_normal = np.concatenate((data_normal, energies[:int(len(energies)/2)]), axis=0)
        data_atk = np.concatenate((data_atk, energies[int(len(energies)/2):]), axis=0)
        cutoff += np.load("Cross_validation/Discretized/Sets{}/cutoff.npy".format(i))

    cutoff = cutoff/10
    weights = np.array([1]*len(data_normal))/len(data_normal)
    bins = np.histogram(np.hstack((data_normal,data_atk)), bins=30)[1]
    plt.xlim(0,350)
    plt.ylim(0,.6)
    plt.hist(data_atk, bins, weights=weights, facecolor=color_attack, alpha=0.7, label="suspicious",  ec='white', linewidth=0.3)
    plt.hist(data_normal, bins, weights=weights, facecolor=color_normal, alpha=0.7, label="unknown",  ec='white', linewidth=0.3)
    plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
    plt.legend()
    plt.xlabel("Energy", fontsize=12)
    plt.ylabel("Probability", fontsize=12)

    plt.savefig("External_test/Results/efc_energies.png", format="png")
    plt.show()

plot_energies_CV()
plot_energies_external()
