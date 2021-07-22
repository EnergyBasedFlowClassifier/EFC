import pandas as pd
import numpy as np
import time
from classification_functions import *
from sklearn.metrics import classification_report, f1_score

sets=1

test = pd.read_csv("5-fold_sets/Discretized/Sets{}/test.csv".format(sets), header=None).astype('int')
test_labels = pd.read_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(sets), squeeze=True, header=None).astype('int')

train = pd.read_csv("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(sets), header=None).astype('int')
train_labels = pd.read_csv("5-fold_sets/Discretized/Sets{}/reduced_train_labels.csv".format(sets), squeeze=True, header=None).astype('int')
print(np.unique(train_labels))


Q = 32
LAMBDA = 0.5


h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(np.array(train), np.array(train_labels), Q, LAMBDA)
predicted = MultiClassPredict(np.array(test), h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))


y_pred = np.load("5-fold_sets/Results/Sets{}/EFC_predicted.npy".format(sets), allow_pickle=True)
print(f1_score(test_labels, y_pred, labels=np.unique(test_labels), average=None))


print(f1_score(test_labels, predicted, labels=np.unique(test_labels), average=None))
