import numpy as np
import pandas as pd
from classification_functions import *
from generic_discretize import *
from sklearn.metrics import classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer(as_frame=True) # load toy dataset from scikit
y = data.target
intervals, X = get_intervals(data.data, 10) # discretize dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False) # split dataset into training and testing sets

idx_abnormal = np.where(y_train == 1)[0] # find abnormal samples indexes in the training set
X_train.drop(idx_abnormal, axis=0, inplace=True) # remove abnormal samples from training
y_train.drop(idx_abnormal, axis=0, inplace=True) # remove the corresponding abonrmal training targets


Q = 11 # max value in dataset (number of bins used to discretize plus one)
LAMBDA = 0.5 # lambda parameter

coupling, h_i, cutoff = create_oneclass_model(np.array(X_train), Q, LAMBDA) # train model
y_predicted, energies = test_oneclass_model(np.array(X_test), coupling, h_i, np.array(y_test), cutoff, Q) # test model
report = classification_report(np.array(y_test), y_predicted) # colect results
print(report)
