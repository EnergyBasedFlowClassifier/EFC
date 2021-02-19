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

def KNN(sets):
    train = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train.npy".format(sets), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.npy".format(sets), allow_pickle=True)
    test = np.load("5-fold_sets/Non_discretized/Sets{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("5-fold_sets/Non_discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)

    KNN = KNeighborsClassifier()
    start = time.time()
    KNN.fit(train, train_labels)
    print("KNN train: ", time.time()-start)
    start = time.time()
    predict_labels = KNN.predict(test)
    print("KNN test: ", time.time()-start)
    np.save("5-fold_sets/Results/Sets{}/KNN_predicted.npy".format(sets), predict_labels)


def RF(sets):
    train = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train.npy".format(sets), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.npy".format(sets), allow_pickle=True)
    test = np.load("5-fold_sets/Non_discretized/Sets{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("5-fold_sets/Non_discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)

    RF = RandomForestClassifier()
    start = time.time()
    RF.fit(train, train_labels)
    print("RF train: ", time.time()-start)
    start = time.time()
    predict_labels = RF.predict(test)
    print("RF test: ", time.time()-start)
    np.save("5-fold_sets/Results/Sets{}/RF_predicted.npy".format(sets), predict_labels)


def GaussianNaiveB(sets):
    train = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train.npy".format(sets), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.npy".format(sets), allow_pickle=True)
    test = np.load("5-fold_sets/Non_discretized/Sets{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("5-fold_sets/Non_discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)

    NB = GaussianNB()
    start = time.time()
    NB.fit(train, train_labels)
    print("NB train: ", time.time()-start)
    start = time.time()
    predict_labels = NB.predict(test)
    print("NB test: ", time.time()-start)
    np.save("5-fold_sets/Results/Sets{}/NB_predicted.npy".format(sets), predict_labels)


def DT(sets):
    train = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train.npy".format(sets), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.npy".format(sets), allow_pickle=True)
    test = np.load("5-fold_sets/Non_discretized/Sets{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("5-fold_sets/Non_discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    print("DT train: ", time.time()-start)
    start = time.time()
    predict_labels = DT.predict(test)
    print("DT test: ", time.time()-start)
    np.save("5-fold_sets/Results/Sets{}/DT_predicted.npy".format(sets), predict_labels)
    print("DT", classification_report(test_labels, predict_labels, labels=np.unique(test_labels)))


def Adaboost(sets):
    train = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train.npy".format(sets), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.npy".format(sets), allow_pickle=True)
    test = np.load("5-fold_sets/Non_discretized/Sets{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("5-fold_sets/Non_discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)

    AD = AdaBoostClassifier()
    start = time.time()
    AD.fit(train, train_labels)
    print("AD train: ", time.time()-start)
    start = time.time()
    predict_labels = AD.predict(test)
    print("AD test: ", time.time()-start)
    np.save("5-fold_sets/Results/Sets{}/AD_predicted.npy".format(sets), predict_labels)

def svc(sets):
    train = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train.npy".format(sets), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.npy".format(sets), allow_pickle=True)
    test = np.load("5-fold_sets/Non_discretized/Sets{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("5-fold_sets/Non_discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)

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
    np.save("5-fold_sets/Results/Sets{}/SVC_predicted.npy".format(sets), predict_labels)


def mlp(sets):
    train = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train.npy".format(sets), allow_pickle=True)
    train_labels = np.load("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.npy".format(sets), allow_pickle=True)
    test = np.load("5-fold_sets/Non_discretized/Sets{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("5-fold_sets/Non_discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)

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
    np.save("5-fold_sets/Results/Sets{}/MLP_predicted.npy".format(sets), predict_labels)


def EFC(sets):
    test = np.load("5-fold_sets/Discretized/Sets{}/test.npy".format(sets), allow_pickle=True).astype('int')
    test_labels = np.load("5-fold_sets/Discretized/Sets{}/test_labels.npy".format(sets)).astype('int')
    train = np.load("5-fold_sets/Discretized/Sets{}/reduced_train.npy".format(sets), allow_pickle=True).astype('int')
    train_labels = np.load("5-fold_sets/Discretized/Sets{}/reduced_train_labels.npy".format(sets)).astype('int')

    Q = 32
    LAMBDA = 0.5

    h_i_matrices, coupling_matrices, cutoffs_list = create_multiclass_model_parallel(train, train_labels, Q, LAMBDA)
    np.save("5-fold_sets/Discretized/Sets{}/h_i_matrices.npy".format(sets), h_i_matrices)
    np.save("5-fold_sets/Discretized/Sets{}/coupling_matrices.npy".format(sets), coupling_matrices)
    np.save("5-fold_sets/Discretized/Sets{}/cutoffs_list.npy".format(sets), cutoffs_list)
    # h_i_matrices = np.load("5-fold_sets/Discretized/Sets{}/h_i_matrices.npy".format(sets), allow_pickle=True)
    # coupling_matrices =  np.load("5-fold_sets/Discretized/Sets{}/coupling_matrices.npy".format(sets), allow_pickle=True)
    # cutoffs_list =  np.load("5-fold_sets/Discretized/Sets{}/cutoffs_list.npy".format(sets), allow_pickle=True)

    print(test.shape)
    print(h_i_matrices.shape)
    print(coupling_matrices.shape)
    print(cutoffs_list.shape)
    print(train_labels.shape)


    predicted = test_multiclass_model_parallel(test, h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))
    # profile = line_profiler.LineProfiler(test_multiclass_model)
    # profile.runcall(test_multiclass_model, test, h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))
    # profile.print_stats()
    # profile.dump_stats("profile_stats.dmp")

    np.save("5-fold_sets/Results/Sets{}/EFC_predicted.npy".format(sets), predicted)
    print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))

for sets in range(1,6):
    EFC(sets)
    mlp(sets)
    KNN(sets)
    GaussianNaiveB(sets)
    DT(sets)
    svc(sets)
    RF(sets)
    Adaboost(sets)
