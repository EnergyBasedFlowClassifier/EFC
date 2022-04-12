import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import sys

sys.path.append("../../../EFC")
from classification_functions import *
import time
from concurrent.futures import ProcessPoolExecutor
import itertools


def svc(args):
    removed, sets = args
    print("SVC", removed, sets)
    train = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/X_train".format(sets), header=None
    )
    train_labels = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/y_train".format(sets), header=None, squeeze=True
    ).squeeze()
    test = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_test".format(sets), header=None)

    valid_indexes = np.where(train_labels == removed)[0]
    train.drop(valid_indexes, axis=0, inplace=True)
    train_labels.drop(valid_indexes, axis=0, inplace=True)

    svc = SVC()
    start = time.time()
    svc.fit(train, train_labels)
    train_time = time.time() - start
    start = time.time()
    predict_labels = svc.predict(test)
    test_time = time.time() - start
    print("svc test: ", time.time() - start)
    print(train_time, test_time)
    np.save(
        "5-fold_sets/Results_removing{}/Sets{}/SVC_predicted.npy".format(removed, sets),
        predict_labels,
    )
    np.save(
        "5-fold_sets/Results_removing{}/Sets{}/SVC_times.npy".format(removed, sets),
        [train_time, test_time],
    )


def mlp(args):
    removed, sets = args
    print("MLP", removed, sets)
    train = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/X_train".format(sets), header=None
    )
    train_labels = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/y_train".format(sets), header=None
    ).squeeze()
    test = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_test".format(sets), header=None)

    valid_indexes = np.where(train_labels == removed)[0]
    train.drop(valid_indexes, axis=0, inplace=True)
    train_labels.drop(valid_indexes, axis=0, inplace=True)

    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time() - start
    print("MLP train: ", training_time)
    start = time.time()
    predict_labels = MLP.predict(test)
    testing_time = time.time() - start
    print("MLP test: ", testing_time)

    np.save(
        "5-fold_sets/Results_removing{}/Sets{}/MLP_predicted.npy".format(removed, sets),
        predict_labels,
    )
    np.save(
        "5-fold_sets/Results_removing{}/Sets{}/MLP_times.npy".format(removed, sets),
        [training_time, testing_time],
    )


def DT(args):
    removed, sets = args
    print("DT", removed, sets)
    train = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/X_train".format(sets), header=None
    )
    train_labels = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/y_train".format(sets), header=None
    ).squeeze()
    test = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_test".format(sets), header=None)

    valid_indexes = np.where(train_labels == removed)[0]
    train.drop(valid_indexes, axis=0, inplace=True)
    train_labels.drop(valid_indexes, axis=0, inplace=True)

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time() - start
    print("DT train: ", time.time() - start)
    start = time.time()
    predict_labels = DT.predict(test)
    testing_time = time.time() - start
    print("DT test: ", time.time() - start)
    np.save(
        "5-fold_sets/Results_removing{}/Sets{}/DT_predicted.npy".format(removed, sets),
        predict_labels,
    )
    np.save(
        "5-fold_sets/Results_removing{}/Sets{}/DT_times.npy".format(removed, sets),
        [training_time, testing_time],
    )


def EFC(args):
    removed, sets = args
    print("EFC", removed, sets)
    test = pd.read_csv(
        "5-fold_sets/Discretized/Sets{}/X_test".format(sets), header=None
    ).astype("int")
    test_labels = pd.read_csv(
        "5-fold_sets/Discretized/Sets{}/y_test".format(sets), squeeze=True, header=None
    ).astype("int")
    train = pd.read_csv(
        "5-fold_sets/Discretized/Sets{}/X_train".format(sets), header=None
    ).astype("int")
    train_labels = pd.read_csv(
        "5-fold_sets/Discretized/Sets{}/y_train".format(sets), squeeze=True, header=None
    ).astype("int")

    valid_indexes = np.where(train_labels == removed)[0]
    train.drop(valid_indexes, axis=0, inplace=True)
    train_labels.drop(valid_indexes, axis=0, inplace=True)

    Q = 30
    LAMBDA = 0.5

    start = time.time()
    h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(
        np.array(train), np.array(train_labels), Q, LAMBDA
    )
    train_time = time.time() - start
    start = time.time()
    predicted, energies = MultiClassPredict(
        np.array(test),
        h_i_matrices,
        coupling_matrices,
        cutoffs_list,
        Q,
        np.unique(train_labels),
    )
    test_time = time.time() - start
    print(train_time, test_time)
    np.save(
        "5-fold_sets/Results_removing{}/Sets{}/EFC_predicted.npy".format(removed, sets),
        predicted,
    )
    print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))
    np.save(
        "5-fold_sets/Results_removing{}/Sets{}/EFC_times.npy".format(removed, sets),
        [train_time, test_time],
    )


lista = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]]
with ProcessPoolExecutor() as executor:
    for subset in lista:
        executor.map(mlp, itertools.product(subset, range(1, 6)))
        executor.map(DT, itertools.product(subset, range(1, 6)))
        executor.map(svc, itertools.product(subset, range(1, 6)))
        executor.map(EFC, itertools.product(subset, range(1, 6)))
