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

def KNN():
    train = pd.read_csv("Data/Normalized/X_train", header=None).astype('int')
    train_labels =  pd.read_csv("Data/Normalized/y_train", squeeze=True, header=None).astype('int')
    test = pd.read_csv("Data/Normalized/X_test", header=None).astype('int')
    test_labels =  pd.read_csv("Data/Normalized/y_test", squeeze=True, header=None).astype('int')

    KNN = KNeighborsClassifier(algorithm='kd_tree',n_jobs=-1)
    start = time.time()
    KNN.fit(train, train_labels)
    training_time = time.time()-start
    print("KNN train: ", time.time()-start)
    start = time.time()
    predict_labels = KNN.predict(test)
    testing_time = time.time()-start
    print("KNN test: ", time.time()-start)
    np.save("Results/KNN_predicted.npy", predict_labels)
    np.save("Results/KNN_times.npy", [training_time, testing_time])



def RF():
    train = pd.read_csv("Data/Normalized/X_train", header=None).astype('int')
    train_labels =  pd.read_csv("Data/Normalized/y_train", squeeze=True, header=None).astype('int')
    test = pd.read_csv("Data/Normalized/X_test", header=None).astype('int')
    test_labels =  pd.read_csv("Data/Normalized/y_test", squeeze=True, header=None).astype('int')

    RF = RandomForestClassifier(n_jobs=-1)
    start = time.time()
    RF.fit(train, train_labels)
    training_time = time.time()-start
    print("RF train: ", time.time()-start)
    start = time.time()
    predict_labels = RF.predict(test)
    testing_time = time.time()-start
    print("RF test: ", time.time()-start)
    np.save("Results/RF_predicted.npy", predict_labels)
    np.save("Results/RF_times.npy", [training_time, testing_time])



def GaussianNaiveB():
    train = pd.read_csv("Data/Normalized/X_train", header=None).astype('int')
    train_labels =  pd.read_csv("Data/Normalized/y_train", squeeze=True, header=None).astype('int')
    test = pd.read_csv("Data/Normalized/X_test", header=None).astype('int')
    test_labels =  pd.read_csv("Data/Normalized/y_test", squeeze=True, header=None).astype('int')

    NB = GaussianNB()
    start = time.time()
    NB.fit(train, train_labels)
    training_time = time.time()-start
    print("NB train: ", time.time()-start)
    start = time.time()
    predict_labels = NB.predict(test)
    testing_time = time.time()-start
    print("NB test: ", time.time()-start)
    np.save("Results/NB_predicted.npy", predict_labels)
    np.save("Results/NB_times.npy", [training_time, testing_time])



def DT():
    train = pd.read_csv("Data/Normalized/X_train", header=None).astype('int')
    train_labels =  pd.read_csv("Data/Normalized/y_train", squeeze=True, header=None).astype('int')
    test = pd.read_csv("Data/Normalized/X_test", header=None).astype('int')
    test_labels =  pd.read_csv("Data/Normalized/y_test", squeeze=True, header=None).astype('int')

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time()-start
    print("DT train: ", time.time()-start)
    start = time.time()
    predict_labels = DT.predict(test)
    testing_time = time.time()-start
    print("DT test: ", time.time()-start)
    np.save("Results/DT_predicted.npy", predict_labels)
    np.save("Results/DT_times.npy", [training_time, testing_time])

    print("DT", classification_report(test_labels, predict_labels, labels=np.unique(test_labels)))


def Adaboost():
    train = pd.read_csv("Data/Normalized/X_train", header=None).astype('int')
    train_labels =  pd.read_csv("Data/Normalized/y_train", squeeze=True, header=None).astype('int')
    test = pd.read_csv("Data/Normalized/X_test", header=None).astype('int')
    test_labels =  pd.read_csv("Data/Normalized/y_test", squeeze=True, header=None).astype('int')

    AD = AdaBoostClassifier()
    start = time.time()
    AD.fit(train, train_labels)
    training_time = time.time()-start
    print("AD train: ", training_time)
    start = time.time()
    predict_labels = AD.predict(test)
    testing_time = time.time()-start
    print("AD test: ", testing_time)
    np.save("Results/AD_predicted.npy", predict_labels)
    np.save("Results/AD_times.npy", [training_time, testing_time])


def svc():
    train = pd.read_csv("Data/Normalized/X_train", header=None).astype('int')
    train_labels =  pd.read_csv("Data/Normalized/y_train", squeeze=True, header=None).astype('int')
    test = pd.read_csv("Data/Normalized/X_test", header=None).astype('int')
    test_labels =  pd.read_csv("Data/Normalized/y_test", squeeze=True, header=None).astype('int')


    svc = SVC(kernel='poly')
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time()-start
    print("SVC train: ", training_time)
    start = time.time()
    predict_labels = svc.predict(test)
    testing_time = time.time()-start
    print("SVC test: ", testing_time)
    np.save("Results/SVC_predicted.npy", predict_labels)
    np.save("Results/SVC_times.npy", [training_time, testing_time])



def mlp():
    train = pd.read_csv("Data/Normalized/X_train", header=None).astype('int')
    train_labels =  pd.read_csv("Data/Normalized/y_train", squeeze=True, header=None).astype('int')
    test = pd.read_csv("Data/Normalized/X_test", header=None).astype('int')
    test_labels =  pd.read_csv("Data/Normalized/y_test", squeeze=True, header=None).astype('int')


    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time()-start
    print("MLP train: ", training_time)
    start = time.time()
    predict_labels = MLP.predict(test)
    testing_time = time.time()-start
    print("MLP test: ", testing_time)

    np.save("Results/MLP_predicted.npy", predict_labels)
    np.save("Results/MLP_times.npy", [training_time, testing_time])



def EFC():
    train = pd.read_csv("Data/Normalized-Discretized/X_train", header=None).astype('int')
    train_labels =  pd.read_csv("Data/Normalized-Discretized/y_train", squeeze=True, header=None).astype('int')
    test = pd.read_csv("Data/Normalized-Discretized/X_test", header=None).astype('int')
    test_labels =  pd.read_csv("Data/Normalized-Discretized/y_test", squeeze=True, header=None).astype('int')
    Q = 66
    LAMBDA = 0.99

    start = time.time()
    h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(np.array(train), np.array(train_labels), Q, LAMBDA)
    training_time = time.time()-start
    print("EFC train: ", training_time)

    start = time.time()
    predicted = MultiClassPredict(np.array(test), h_i_matrices, coupling_matrices, cutoffs_list, Q, np.unique(train_labels))
    testing_time = time.time()-start
    print("EFC test: ", testing_time)

    np.save("Results/EFC_predicted.npy", predicted)
    np.save("Results/EFC_times.npy", [training_time, testing_time])

    print(classification_report(test_labels, predicted, labels=np.unique(test_labels)))



# EFC()
# mlp()
# KNN()
# GaussianNaiveB()
# DT()
svc()
# RF()
