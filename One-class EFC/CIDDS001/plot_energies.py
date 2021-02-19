import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

color_attack = "#006680"
color_normal = "#b3b3b3"
data_normal = np.array([])
data_atk = np.array([])
cutoff = 0
for set in range(1,11):
    energies = np.load("Data/Discretized/Exp{}/energies_internal.npy".format(set), allow_pickle=True)
    data_normal = np.concatenate((data_normal, energies[:len(energies)//2]),axis=0)
    data_atk = np.concatenate((data_atk, energies[len(energies)//2::]), axis=0)
    cutoff += np.load("Data/Discretized/Exp{}/cutoff.npy".format(set), allow_pickle=True)
cutoff /= 10
weights = np.array([1]*len(data_normal))/len(data_normal)
bins = np.histogram(np.hstack((data_normal,data_atk)), bins=30)[1]
plt.hist(data_atk, bins, weights=weights, facecolor=color_attack, alpha=0.7, ec='white', linewidth=0.3, label="malicious")
plt.hist(data_normal, bins, weights=weights, facecolor=color_normal, alpha=0.7, ec='white', linewidth=0.3, label="benign")
plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
plt.ylim(0,0.4)
plt.xlim(0,300)
plt.legend()
plt.xlabel("Energy", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.savefig("Data/Results/CIDDS001_energies_internal.png", format="png")
plt.show()


color_attack = "#006680"
color_normal = "#b3b3b3"
data_normal = np.array([])
data_atk = np.array([])
cutoff = 0
for set in range(1,11):
    energies = np.load("Data/Discretized/Exp{}/energies_external.npy".format(set), allow_pickle=True)
    data_normal = np.concatenate((data_normal, energies[:len(energies)//2]),axis=0)
    data_atk = np.concatenate((data_atk, energies[len(energies)//2::]), axis=0)
    cutoff += np.load("Data/Discretized/Exp{}/cutoff.npy".format(set), allow_pickle=True)
cutoff /= 10
weights = np.array([1]*len(data_normal))/len(data_normal)
bins = np.histogram(np.hstack((data_normal,data_atk)), bins=30)[1]
plt.hist(data_atk, bins, weights=weights, facecolor=color_attack, alpha=0.7, ec='white', linewidth=0.3, label="suspicious")
plt.hist(data_normal, bins, weights=weights, facecolor=color_normal, alpha=0.7, ec='white', linewidth=0.3, label="unknown")
plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
plt.ylim(0,0.4)
plt.xlim(0,300)
plt.legend()
plt.xlabel("Energy", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.savefig("Data/Results/CIDDS001_energies_external.png", format="png")
plt.show()
