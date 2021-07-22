import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC, OneClassSVM
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from classification_functions import *
import pickle
import os
import time

def KNN(sets, exp):
    train = np.load("TimeData/Non_discretized/Size{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("TimeData/Non_discretized/Size{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("TimeData/Non_discretized/Size{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("TimeData/Non_discretized/Size{}/train_labels.npy".format(sets), allow_pickle=True)

    KNN = KNeighborsClassifier(algorithm='kd_tree')
    start = time.time()
    KNN.fit(train, train_labels)
    training_time = time.time()-start

    start = time.time()
    predict_labels_internal = KNN.predict(test)
    testing_time = time.time()-start
    print("Train:", training_time)
    print("Test:", testing_time)

    np.save("TimeData/Results/Size{}/Exp{}/KNN_times.npy".format(sets, exp), [training_time, testing_time])


def RF(sets, exp):
    train = np.load("TimeData/Non_discretized/Size{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("TimeData/Non_discretized/Size{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("TimeData/Non_discretized/Size{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("TimeData/Non_discretized/Size{}/train_labels.npy".format(sets), allow_pickle=True)

    RF = RandomForestClassifier()
    start = time.time()
    RF.fit(train, train_labels)
    training_time = time.time()-start


    start = time.time()
    predict_labels_internal = RF.predict(test)
    testing_time = time.time()-start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save("TimeData/Results/Size{}/Exp{}/RF_times.npy".format(sets, exp), [training_time, testing_time])



def GaussianNaiveB(sets, exp):
    train = np.load("TimeData/Non_discretized/Size{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("TimeData/Non_discretized/Size{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("TimeData/Non_discretized/Size{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("TimeData/Non_discretized/Size{}/train_labels.npy".format(sets), allow_pickle=True)

    NB = GaussianNB()
    start = time.time()
    NB.fit(train, train_labels)
    training_time = time.time()-start

    start = time.time()
    predict_labels_internal = NB.predict(test)
    testing_time = time.time()-start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save("TimeData/Results/Size{}/Exp{}/GaussianNB_times.npy".format(sets, exp), [training_time, testing_time])

def DT(sets, exp):
    train = np.load("TimeData/Non_discretized/Size{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("TimeData/Non_discretized/Size{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("TimeData/Non_discretized/Size{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("TimeData/Non_discretized/Size{}/train_labels.npy".format(sets), allow_pickle=True)

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time()-start


    start = time.time()
    predict_labels_internal = DT.predict(test)
    testing_time = time.time()-start
    print("DT train:", training_time)
    print("DT test:", testing_time)
    np.save("TimeData/Results/Size{}/Exp{}/DT_times.npy".format(sets, exp), [training_time, testing_time])

def Adaboost(sets, exp):
    train = np.load("TimeData/Non_discretized/Size{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("TimeData/Non_discretized/Size{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("TimeData/Non_discretized/Size{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("TimeData/Non_discretized/Size{}/train_labels.npy".format(sets), allow_pickle=True)

    AD = AdaBoostClassifier()
    start = time.time()
    AD.fit(train, train_labels)
    training_time = time.time()-start


    start = time.time()
    predict_labels_internal = AD.predict(test)
    testing_time = time.time()-start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save("TimeData/Results/Size{}/Exp{}/Adaboost_times.npy".format(sets, exp), [training_time, testing_time])


def svc(sets, exp):
    train = np.load("TimeData/Non_discretized/Size{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("TimeData/Non_discretized/Size{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("TimeData/Non_discretized/Size{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("TimeData/Non_discretized/Size{}/train_labels.npy".format(sets), allow_pickle=True)

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)


    svc = SVC(kernel='poly', probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time()-start

    start = time.time()
    predict_labels_internal = svc.predict(test)
    testing_time = time.time()-start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save("TimeData/Results/Size{}/Exp{}/SVC_times.npy".format(sets, exp), [training_time, testing_time])

def mlp(sets, exp):
    train = np.load("TimeData/Non_discretized/Size{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("TimeData/Non_discretized/Size{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("TimeData/Non_discretized/Size{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("TimeData/Non_discretized/Size{}/train_labels.npy".format(sets), allow_pickle=True)

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)


    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time()-start

    start = time.time()
    predict_labels_internal = MLP.predict(test)
    testing_time = time.time()-start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save("TimeData/Results/Size{}/Exp{}/MLP_times.npy".format(sets, exp), [training_time, testing_time])

def EFC(sets, exp):
    train = np.load("TimeData/Discretized/Size{}/train.npy".format(sets), allow_pickle=True).astype('int')
    train_normal = train[:int(train.shape[0]/2), :]
    test = np.load("TimeData/Discretized/Size{}/test.npy".format(sets), allow_pickle=True).astype('int')
    test_labels = np.load("TimeData/Discretized/Size{}/test_labels.npy".format(sets), allow_pickle=True).astype('int')

    Q = 32
    LAMBDA = 0.5

    # Creating model
    start = time.time()
    couplingmatrix, h_i, cutoff = create_oneclass_model(train_normal, Q, LAMBDA)
    training_time = time.time()-start

    start = time.time()
    predicted_labels_internal, energies_internal = test_oneclass_model(np.array(test,dtype=int), couplingmatrix, h_i, test_labels, CUTOFF, Q)
    testing_time = time.time()-start
    np.save("TimeData/Results/Size{}/Exp{}/EFC_times.npy".format(sets, exp), [training_time, testing_time])

    print("Train:", training_time)
    print("Test:", testing_time)

for size in [0,1,2,3]:
    for exp in range(1,11):
        os.makedirs("TimeData/Results/Size{}/Exp{}/".format(size, exp), exist_ok=True)
        EFC(size, exp)
        KNN(size, exp)
        RF(size, exp)
        DT(size, exp)
        Adaboost(size, exp)
        GaussianNaiveB(size, exp)
        svc(size, exp)
        mlp(size, exp)
