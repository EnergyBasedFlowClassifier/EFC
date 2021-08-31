import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, recall_score
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import sys
sys.path.append('../../../efc')
from classification_functions import *
import time

train = pd.read_csv("Data/Normalized-Discretized/X_train", header=None).astype('int')
train_labels =  pd.read_csv("Data/Normalized-Discretized/y_train", squeeze=True, header=None).astype('int')
test = pd.read_csv("Data/Normalized-Discretized/X_test", header=None).astype('int')
test_labels =  pd.read_csv("Data/Normalized-Discretized/y_test", squeeze=True, header=None).astype('int')

Q = 66
LAMBDA = 0.99

h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(np.array(train), np.array(train_labels), Q, LAMBDA)
predicted = MultiClassPredict(np.array(test), h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))

np.save("predicted", predicted)

plt.rcParams.update({'font.size': 14})
names = ['Normal','DoS','Probe', 'R2L', 'U2R', 'Unknown']

cf = pd.DataFrame(confusion_matrix(test_labels, predicted, normalize='true'), index = names, columns = names)
sns.heatmap(cf, annot=True, cmap="Blues", fmt='.2f')
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.yticks(rotation=30)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("CM_Test.pdf", format='pdf',bbox_inches = "tight")
plt.show()

test_unknown_unique = pd.read_csv("Data/Normalized-Discretized/X_test_unknown_unique", header=None).astype('int')
test_labels_unknown_unique =  pd.read_csv("Data/Normalized-Discretized/y_test_unknown_unique", squeeze=True, header=None).astype('int')
predicted_unknown_unique = MultiClassPredict(np.array(test_unknown_unique), h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))
np.save("predicted_unknown_unique", predicted_unknown_unique)

cf = pd.DataFrame(confusion_matrix(test_labels_unknown_unique, predicted_unknown_unique, normalize='true'), index = names, columns = names)
sns.heatmap(cf, annot=True, cmap="Blues", fmt='.2f')
plt.ylabel("True label")
plt.xlabel("Predicted label")
plt.yticks(rotation=30)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("CM_Test_unique.pdf", format='pdf',bbox_inches = "tight")
plt.show()
