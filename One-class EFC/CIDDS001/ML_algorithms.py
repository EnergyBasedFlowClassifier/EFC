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

def KNN(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)

    external_test = np.load("External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    KNN = KNeighborsClassifier()
    start = time.time()
    KNN.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = KNN.predict_proba(test)
    start = time.time()
    predict_labels_internal = KNN.predict(test)
    testing_time = time.time()-start

    np.save("Data/Results/Exp{}/KNN_times.npy".format(sets), [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/KNN_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = KNN.predict_proba(external_test)
    predict_labels_external = KNN.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/KNN_external.npy".format(sets), [precision, recall, f1, roc])



def RF(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)

    external_test = np.load("External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    RF = RandomForestClassifier()
    start = time.time()
    RF.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = RF.predict_proba(test)
    start = time.time()
    predict_labels_internal = RF.predict(test)
    testing_time = time.time()-start
    np.save("Data/Results/Exp{}/RF_times.npy".format(sets), [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/RF_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = RF.predict_proba(external_test)
    predict_labels_external = RF.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/RF_external.npy".format(sets), [precision, recall, f1, roc])



def GaussianNaiveB(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)

    external_test = np.load("External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    NB = GaussianNB()
    start = time.time()
    NB.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = NB.predict_proba(test)
    start = time.time()
    predict_labels_internal = NB.predict(test)
    testing_time = time.time()-start

    np.save("Data/Results/Exp{}/GaussianNB_times.npy".format(sets), [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/GaussianNB_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = NB.predict_proba(external_test)
    predict_labels_external = NB.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/GaussianNB_external.npy".format(sets), [precision, recall, f1, roc])

def DT(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)

    external_test = np.load("External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time()-start


    predict_prob_internal = DT.predict_proba(test)
    start = time.time()
    predict_labels_internal = DT.predict(test)
    testing_time = time.time()-start
    np.save("Data/Results/Exp{}/DT_times.npy".format(sets), [training_time, testing_time])

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/DT_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = DT.predict_proba(external_test)
    predict_labels_external = DT.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/DT_external.npy".format(sets), [precision, recall, f1, roc])

def Adaboost(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)

    external_test = np.load("External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    AD = AdaBoostClassifier()
    start = time.time()
    AD.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = AD.predict_proba(test)
    start = time.time()
    predict_labels_internal = AD.predict(test)
    testing_time = time.time()-start
    np.save("Data/Results/Exp{}/Adaboost_times.npy".format(sets), [training_time, testing_time])

    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/Adaboost_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = AD.predict_proba(external_test)
    predict_labels_external = AD.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/Adaboost_external.npy".format(sets), [precision, recall, f1, roc])

def svc(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)

    external_test = np.load("External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)
    transformer = Normalizer().fit(external_test)
    external_test = transformer.transform(external_test)

    svc = SVC(kernel='poly', probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = svc.predict_proba(test)
    start = time.time()
    predict_labels_internal = svc.predict(test)
    testing_time = time.time()-start

    np.save("Data/Results/Exp{}/SVC_times.npy".format(sets), [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/SVC_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = svc.predict_proba(external_test)
    predict_labels_external = svc.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/SVC_external.npy".format(sets), [precision, recall, f1, roc])

def mlp(sets):
    train = np.load("Data/Non_discretized/Exp{}/train.npy".format(sets), allow_pickle=True)
    test = np.load("Data/Non_discretized/Exp{}/test.npy".format(sets), allow_pickle=True)
    test_labels = np.load("Data/Non_discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True)
    train_labels = np.load("Data/Non_discretized/Exp{}/train_labels.npy".format(sets), allow_pickle=True)

    external_test = np.load("External_test/Non_discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True)
    external_test_labels = np.load("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True)

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)
    transformer = Normalizer().fit(external_test)
    external_test = transformer.transform(external_test)

    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time()-start

    predict_prob_internal = MLP.predict_proba(test)
    start = time.time()
    predict_labels_internal = MLP.predict(test)
    testing_time = time.time()-start
    np.save("Data/Results/Exp{}/MLP_times.npy".format(sets), [training_time, testing_time])
    precision = precision_score(test_labels, predict_labels_internal)
    recall = recall_score(test_labels, predict_labels_internal)
    f1 = f1_score(test_labels, predict_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob_internal[:,1])
    np.save("Data/Results/Exp{}/MLP_internal.npy".format(sets), [precision, recall, f1, roc])

    predict_prob_external = MLP.predict_proba(external_test)
    predict_labels_external = MLP.predict(external_test)
    precision = precision_score(external_test_labels, predict_labels_external)
    recall = recall_score(external_test_labels, predict_labels_external)
    f1 = f1_score(external_test_labels, predict_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob_external[:,1])
    np.save("Data/Results/Exp{}/MLP_external.npy".format(sets), [precision, recall, f1, roc])

def EFC(sets):
    train = np.load("Data/Discretized/Exp{}/train.npy".format(sets), allow_pickle=True).astype('int')
    train_normal = train[:int(train.shape[0]/2), :]
    test = np.load("Data/Discretized/Exp{}/test.npy".format(sets), allow_pickle=True).astype('int')
    test_labels = np.load("Data/Discretized/Exp{}/test_labels.npy".format(sets), allow_pickle=True).astype('int')

    external_test = np.load("External_test/Discretized/Exp{}/external_test.npy".format(sets), allow_pickle=True).astype('int')
    external_test_labels = np.load("External_test/Discretized/Exp{}/external_test_labels.npy".format(sets), allow_pickle=True).astype('int')

    Q = 32
    LAMBDA = 0.5

    # Creating model
    start = time.time()
    couplingmatrix, h_i, cutoff = create_oneclass_model(train_normal, Q, LAMBDA)
    training_time = time.time()-start
    np.save("Data/Discretized/Exp{}/cutoff.npy".format(sets), np.array(cutoff))
    np.save("Data/Discretized/Exp{}/h_i.npy".format(sets), h_i)
    np.save("Data/Discretized/Exp{}/couplingmatrix.npy".format(sets), couplingmatrix)

    start = time.time()
    predicted_labels_internal, energies_internal = test_oneclass_model(np.array(test,dtype=int), couplingmatrix, h_i, test_labels, cutoff, Q)
    testing_time = time.time()-start
    np.save("Data/Results/Exp{}/EFC_times.npy".format(sets), [training_time, testing_time])

    print("Train:", training_time)
    print("Test:", testing_time)

    np.save("Data/Discretized/Exp{}/energies_internal.npy".format(sets), np.array(energies_internal))
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies_internal).reshape(-1,1))]
    precision = precision_score(test_labels, predicted_labels_internal)
    recall = recall_score(test_labels, predicted_labels_internal)
    f1 = f1_score(test_labels, predicted_labels_internal)
    roc = roc_auc_score(test_labels, predict_prob)
    np.save("Data/Results/Exp{}/EFC_internal.npy".format(sets), np.array([precision, recall, f1, roc]))
    print(f1, roc)

    predicted_labels_external, energies_external = test_oneclass_model(np.array(external_test,dtype=int), couplingmatrix, h_i, external_test_labels, cutoff, Q)
    np.save("Data/Discretized/Exp{}/energies_external.npy".format(sets), np.array(energies_external))
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies_external).reshape(-1,1))]
    precision = precision_score(external_test_labels, predicted_labels_external)
    recall = recall_score(external_test_labels, predicted_labels_external)
    f1 = f1_score(external_test_labels, predicted_labels_external)
    roc = roc_auc_score(external_test_labels, predict_prob)
    print(f1, roc)
    np.save("Data/Results/Exp{}/EFC_external.npy".format(sets), np.array([precision, recall, f1, roc]))


for i in range(1,11):
    EFC(i)
    KNN(i)
    RF(i)
    DT(i)
    Adaboost(i)
    GaussianNaiveB(i)
    svc(i)
    mlp(i)
