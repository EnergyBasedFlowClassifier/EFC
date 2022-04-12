import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.append("../../../EFC")
from classification_functions import *
import time


def DT(sets):
    train = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/X_train".format(sets), header=None
    )
    train_labels = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/y_train".format(sets), header=None
    )
    test = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_test".format(sets), header=None)

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time() - start
    print("DT train: ", time.time() - start)
    start = time.time()
    predict_labels = DT.predict(test)
    testing_time = time.time() - start
    print("DT test: ", time.time() - start)
    np.save("5-fold_sets/Results/Sets{}/DT_predicted.npy".format(sets), predict_labels)
    np.save(
        "5-fold_sets/Results/Sets{}/DT_times.npy".format(sets),
        [training_time, testing_time],
    )


def svc(sets):
    train = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/X_train".format(sets), header=None
    )
    train_labels = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/y_train".format(sets), header=None
    )
    test = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_test".format(sets), header=None)

    svc = SVC(kernel="poly", probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time() - start
    print("SVC train: ", training_time)
    start = time.time()
    predict_labels = svc.predict(test)
    testing_time = time.time() - start
    print("SVC test: ", testing_time)
    np.save("5-fold_sets/Results/Sets{}/SVC_predicted.npy".format(sets), predict_labels)
    np.save(
        "5-fold_sets/Results/Sets{}/SVC_times.npy".format(sets),
        [training_time, testing_time],
    )


def mlp(sets):
    train = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/X_train".format(sets), header=None
    )
    train_labels = pd.read_csv(
        "5-fold_sets/Normalized/Sets{}/y_train".format(sets), header=None
    )
    test = pd.read_csv("5-fold_sets/Normalized/Sets{}/X_test".format(sets), header=None)

    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time() - start
    print("MLP train: ", training_time)
    start = time.time()
    predict_labels = MLP.predict(test)
    testing_time = time.time() - start
    print("MLP test: ", testing_time)

    np.save("5-fold_sets/Results/Sets{}/MLP_predicted.npy".format(sets), predict_labels)
    np.save(
        "5-fold_sets/Results/Sets{}/MLP_times.npy".format(sets),
        [training_time, testing_time],
    )


def EFC(sets):
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
    Q = 30
    LAMBDA = 0.5
    cutoff = 0.95

    start = time.time()
    h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(
        np.array(train), np.array(train_labels), Q, LAMBDA, cutoff
    )
    training_time = time.time() - start
    print("EFC train: ", training_time)

    start = time.time()
    predicted, energies = MultiClassPredict(
        np.array(test),
        h_i_matrices,
        coupling_matrices,
        cutoffs_list,
        Q,
        np.unique(train_labels),
    )
    testing_time = time.time() - start
    print("EFC test: ", testing_time)

    np.save("5-fold_sets/Results/Sets{}/EFC_predicted.npy".format(sets), predicted)
    np.save(
        "5-fold_sets/Results/Sets{}/EFC_times.npy".format(sets),
        [training_time, testing_time],
    )

    print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))


with ProcessPoolExecutor() as executor:
    executor.map(mlp, range(1, 6))
    executor.map(DT, range(1, 6))
    executor.map(svc, range(1, 6))
    executor.map(EFC, range(1, 6))
