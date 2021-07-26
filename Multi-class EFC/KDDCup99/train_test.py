import sys
sys.path.append('../../../efc')
from classification_functions import *
import pandas as pd
import numpy as np

train = pd.read_csv("X_kddcup.data_10_percent_discretized", header=None).astype('int')
train_labels =  pd.read_csv("y_kddcup.data_10_percent_discretized", squeeze=True, header=None).astype('int')
print(np.unique(train_labels, return_counts=True))
print(train_labels.shape[0])

test = pd.read_csv("X_corrected_discretized", header=None).astype('int')
test_labels =  pd.read_csv("y_corrected_discretized", squeeze=True, header=None).astype('int')
print(np.unique(test_labels))

Q = 66
LAMBDA = 0.5
THETA = 0.9

h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(np.array(train), np.array(train_labels), Q, LAMBDA)

predicted = MultiClassPredict(np.array(test), h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))
np.save("predicted", predicted)
