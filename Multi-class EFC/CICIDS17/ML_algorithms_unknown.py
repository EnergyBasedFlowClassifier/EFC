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
from classification_functions_parallel import *
import time

def KNN(removed, sets):
    train = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(sets), delimiter=',')
    train_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(sets), delimiter=',')
    test = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test.csv".format(sets), delimiter=',')
    test_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(sets), delimiter=',')
    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    train = train[valid_indexes, :]
    train_labels = train_labels[valid_indexes]

    KNN = KNeighborsClassifier()
    start = time.time()
    KNN.fit(train, train_labels)
    print("KNN train: ", time.time()-start)
    start = time.time()
    predict_labels = KNN.predict(test)
    print("KNN test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/KNN_predicted.npy".format(removed, sets), predict_labels)


def RF(removed, sets):
    train = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(sets), delimiter=',')
    train_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(sets), delimiter=',')
    test = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test.csv".format(sets), delimiter=',')
    test_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(sets), delimiter=',')
    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    train = train[valid_indexes, :]
    train_labels = train_labels[valid_indexes]

    RF = RandomForestClassifier()
    start = time.time()
    RF.fit(train, train_labels)
    print("RF train: ", time.time()-start)
    start = time.time()
    predict_labels = RF.predict(test)
    print("RF test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/RF_predicted.npy".format(removed, sets), predict_labels)


def GaussianNaiveB(removed, sets):
    train = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(sets), delimiter=',')
    train_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(sets), delimiter=',')
    test = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test.csv".format(sets), delimiter=',')
    test_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(sets), delimiter=',')
    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    train = train[valid_indexes, :]
    train_labels = train_labels[valid_indexes]

    NB = GaussianNB()
    start = time.time()
    NB.fit(train, train_labels)
    print("NB train: ", time.time()-start)
    start = time.time()
    predict_labels = NB.predict(test)
    print("NB test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/NB_predicted.npy".format(removed, sets), predict_labels)


def DT(removed, sets):
    train = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(sets), delimiter=',')
    train_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(sets), delimiter=',')
    test = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test.csv".format(sets), delimiter=',')
    test_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(sets), delimiter=',')
    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    train = train[valid_indexes, :]
    train_labels = train_labels[valid_indexes]

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    print("DT train: ", time.time()-start)
    start = time.time()
    predict_labels = DT.predict(test)
    print("DT test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/DT_predicted.npy".format(removed, sets), predict_labels)
    print("DT", classification_report(test_labels, predict_labels, labels=np.unique(test_labels)))


def Adaboost(removed, sets):
    train = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(sets), delimiter=',')
    train_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(sets), delimiter=',')
    test = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test.csv".format(sets), delimiter=',')
    test_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(sets), delimiter=',')
    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    train = train[valid_indexes, :]
    train_labels = train_labels[valid_indexes]

    AD = AdaBoostClassifier()
    start = time.time()
    AD.fit(train, train_labels)
    print("AD train: ", time.time()-start)
    start = time.time()
    predict_labels = AD.predict(test)
    print("AD test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/AD_predicted.npy".format(removed, sets), predict_labels)


def svc(removed, sets):
    train = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(sets), delimiter=',')
    train_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(sets), delimiter=',')
    test = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test.csv".format(sets), delimiter=',')
    test_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(sets), delimiter=',')
    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    train = train[valid_indexes, :]
    train_labels = train_labels[valid_indexes]

    transformer = Normalizer().fit(test)
    test = transformer.transform(test)
    transformer = Normalizer().fit(train)
    train = transformer.transform(train)

    svc = SVC(kernel='poly', probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    print("SVC train: ", time.time()-start)
    start = time.time()
    predict_labels = svc.predict(test)
    print("SVC test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/SVC_predicted.npy".format(removed, sets), predict_labels)


def mlp(removed, sets):
    train = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(sets), delimiter=',')
    train_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(sets), delimiter=',')
    test = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test.csv".format(sets), delimiter=',')
    test_labels = np.genfromtxt("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(sets), delimiter=',')
    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    print(train.shape)
    train = train[valid_indexes, :]
    print(train.shape)
    train_labels = train_labels[valid_indexes]
    print(train_labels.shape)
    unique, counts = np.unique(train_labels, return_counts=True)
    print(unique)
    print(counts)

    transformer = Normalizer().fit(test)
    test = transformer.transform(test)
    transformer = Normalizer().fit(train)
    train = transformer.transform(train)

    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    print("MLP train: ", time.time()-start)
    start = time.time()
    predict_labels = MLP.predict(test)
    print("MLP test: ", time.time()-start)
    np.save("5-fold_sets/Results_removing{}/Sets{}/MLP_predicted.npy".format(removed, sets), predict_labels)


def EFC(removed, sets):
    test = np.genfromtxt("5-fold_sets/Discretized/Sets{}/test.csv".format(sets), delimiter=',').astype('int')
    test_labels = np.genfromtxt("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(sets)).astype('int')
    train = np.genfromtxt("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(sets), delimiter=',').astype('int')
    train_labels = np.genfromtxt("5-fold_sets/Discretized/Sets{}/reduced_train_labels.csv".format(sets)).astype('int')
    valid_indexes = [idx for idx, item in enumerate(train_labels) if item != removed]
    print(train.shape)
    train = train[valid_indexes, :]
    print(train.shape)
    train_labels = train_labels[valid_indexes]
    print(train_labels.shape)
    unique, counts = np.unique(train_labels, return_counts=True)
    print(unique)
    print(counts)

    Q = 30
    LAMBDA = 0.5

    h_i_matrices, coupling_matrices, cutoffs_list = create_multiclass_model_parallel(train, train_labels, Q, LAMBDA)

    predicted = test_multiclass_model_parallel(test, h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))
    np.save("5-fold_sets/Results_removing{}/Sets{}/sinal.npy".format(removed, sets), ['sinal'])

    np.save("5-fold_sets/Results_removing{}/Sets{}/EFC_predicted.npy".format(removed, sets), predicted)
    print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))

#,1,2,3,4,5,6,
for removed in [7,8,9,10,11,12]:
    for sets in range(1,6):
        EFC(removed, sets)
        # mlp(removed, sets)
        # KNN(removed, sets)
        # GaussianNaiveB(removed, sets)
        DT(removed, sets)
        # svc(removed, sets)
        # RF(removed, sets)
        # Adaboost(removed, sets)
