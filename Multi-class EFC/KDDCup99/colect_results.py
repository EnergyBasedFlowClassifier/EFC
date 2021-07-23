import numpy as np
import seaborn as sns
import pandas as pd
import os
import pickle
from statistics import mean, stdev
from sklearn.metrics import *
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from scipy.stats import gmean, gstd
import seaborn as sns

names = ['normal','DoS','Probe', 'R2L', 'U2R']
y_true = pd.read_csv("y_corrected_discretized", squeeze=True, header=None).astype('int')
print(np.unique(y_true, return_counts=True))
y_pred = np.load("predicted.npy")
print(classification_report(y_true, y_pred, target_names=['normal','DoS','Probe', 'R2L', 'U2R', 'Unknown']))
cf = pd.DataFrame(confusion_matrix(y_true, y_pred, normalize='true'), index = ['normal','DoS','Probe', 'R2L', 'U2R', 'Unknown'], columns = ['normal','DoS','Probe', 'R2L', 'U2R', 'Unknown'])
sns.heatmap(cf, annot=True, cmap="Blues", fmt='.2f')
plt.ylabel("True label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)
plt.yticks(rotation=45)
plt.savefig("EFC_Confusion_matrix.pdf", format='pdf',bbox_inches = "tight")
