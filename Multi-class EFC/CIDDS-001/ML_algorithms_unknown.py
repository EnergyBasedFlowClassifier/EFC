import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import Normalizer
from sklearn.metrics import classification_report
import pickle
import sys
sys.path.append('../../../efc')
from classification_functions import *
import time

def RF(removed, sets):
    train = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/encoded_train.csv".format(sets), header=None))
    train_labels = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/encoded_train_labels.csv".format(sets), header=None))
    test = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/encoded_test.csv".format(sets), header=None))
    test_labels = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(sets), header=None))

    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    train = train[valid_indexes, :]
    train_labels = train_labels[valid_indexes]

    RF = RandomForestClassifier(n_jobs=-1)
    start = time.time()
    RF.fit(train, train_labels)
    print("RF train: ", time.time()-start)
    start = time.time()
    predict_labels = RF.predict(test)
    print("RF test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/RF_predicted.npy".format(removed, sets), predict_labels)


def EFC(removed, sets):
    test = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/test.csv".format(sets), header=None).astype('int'))
    test_labels = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(sets), squeeze=True, header=None).astype('int'))
    train = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(sets), header=None).astype('int'))
    train_labels = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/reduced_train_labels.csv".format(sets), squeeze=True, header=None).astype('int'))

    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    train = train[valid_indexes, :]
    train_labels = train_labels[valid_indexes]

    Q = 32
    LAMBDA = 0.5

    h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(train, train_labels, Q, LAMBDA)

    predicted = MultiClassPredict(test, h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))

    np.save("5-fold_sets/Results_removing{}/Sets{}/EFC_predicted.npy".format(removed, sets), predicted)
    print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))


for removed in [1, 2, 3, 4]:
    for sets in range(1,6):
        EFC(removed, sets)
        RF(removed, sets)
