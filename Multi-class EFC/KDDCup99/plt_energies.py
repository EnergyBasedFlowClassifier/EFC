import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

y_pred = np.array(pd.read_csv("y_corrected_discretized", squeeze=True, header=None).astype('int'))
y_energies = np.load("energies.npy")
a = y_energies[np.where(y_pred==0)[0]]
b = y_energies[np.where(y_pred==1)[0]]
c = y_energies[np.where(y_pred==2)[0]]
d = y_energies[np.where(y_pred==3)[0]]
e = y_energies[np.where(y_pred==4)[0]]
f = y_energies[np.where(y_pred==100)[0]]


bins = np.histogram(np.hstack((a, b, c, d, e, f)), bins=100)[1]
plt.hist(a, bins, facecolor='b', alpha=0.7, weights=np.array([1]*len(a))/len(a), ec='white', linewidth=0.3, label="normal")
plt.hist(b, bins, facecolor='g', alpha=0.7, weights=np.array([1]*len(b))/len(b), ec='white', linewidth=0.3, label="DoS")
plt.hist(c, bins, facecolor='r', alpha=0.7, weights=np.array([1]*len(c))/len(c), ec='white', linewidth=0.3, label="Probe")
plt.hist(d, bins, facecolor='c', alpha=0.7, weights=np.array([1]*len(d))/len(d), ec='white', linewidth=0.3, label="R2L")
plt.hist(e, bins, facecolor='m', alpha=0.7, weights=np.array([1]*len(e))/len(e), ec='white', linewidth=0.3, label="U2R")
plt.hist(f, bins, facecolor='y', alpha=0.7, weights=np.array([1]*len(f))/len(f), ec='white', linewidth=0.3, label="Unknown")
# plt.axvline(cutoff, color='r', linestyle='dashed', linewidth=1)
plt.legend()
plt.xlabel("Energy", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.savefig("energies.png", format="png")
plt.show()
