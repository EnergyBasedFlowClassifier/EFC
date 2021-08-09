import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import sys
sys.path.append('../../../efc')
from classification_functions import *
import time

train_origin = "Encoded/Normalized-Discretized"

if sys.argv[1] == '0':
    test_origin = "Encoded/Normalized-Discretized"
if sys.argv[1] == '1':
    test_origin = "Encoded/Unique-Unknown-Normalized-Discretized"


train = pd.read_csv("Data/{}/X_train".format(train_origin), header=None).astype('int')
train_labels =  pd.read_csv("Data/{}/y_train".format(train_origin), squeeze=True, header=None).astype('int')
test = pd.read_csv("Data/{}/X_test".format(test_origin), header=None).astype('int')
test_labels =  pd.read_csv("Data/{}/y_test".format(test_origin), squeeze=True, header=None).astype('int')

Q = 87
LAMBDA = 0.9

h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(np.array(train), np.array(train_labels), Q, LAMBDA)
np.save("h_i_matrices.npy", h_i_matrices)
np.save("coupling_matrices.npy", coupling_matrices)
np.save("cutoffs_list.npy", cutoffs_list)


predicted, energies = MultiClassPredict(np.array(test), h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))
np.save("Results/EFC_predicted.npy", predicted)

print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))
names = ['normal','DoS','Probe', 'R2L', 'U2R', 'Unknown']
cf = pd.DataFrame(confusion_matrix(test_labels, predicted, normalize='true'), index = names, columns = names)
sns.heatmap(cf, annot=True, cmap="Blues", fmt='.2f')
plt.ylabel("True label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)
plt.yticks(rotation=45)
plt.savefig("CM_Test_{}.pdf".format(test_origin[8:]), format='pdf',bbox_inches = "tight")
