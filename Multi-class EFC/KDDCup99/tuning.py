import sys
sys.path.append('../../../efc')
from classification_functions import *
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import confusion_matrix, recall_score

train = pd.read_csv("Data/Normalized-Discretized/X_train", header=None).astype('int')
train_labels =  pd.read_csv("Data/Normalized-Discretized/y_train", squeeze=True, header=None).astype('int')
print(np.unique(train_labels, return_counts=True))
print(train_labels.shape[0])

validation = pd.read_csv("Data/Normalized-Discretized/X_validation", header=None).astype('int')
validation_labels =  pd.read_csv("Data/Normalized-Discretized/y_validation", squeeze=True, header=None).astype('int')
print(np.unique(validation_labels, return_counts=True))

Q = 67

for LAMBDA in [0.5, 0.6, 0.7, 0.8, 0.9]:
    h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(np.array(train), np.array(train_labels), Q, LAMBDA)
    predicted, energies = MultiClassPredict(np.array(validation), h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))

    print(LAMBDA, recall_score(validation_labels, predicted, average=None, labels=[0,1,2,3,4]))
