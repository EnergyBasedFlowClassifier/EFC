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
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pickle
from classification_functions import *
from sklearn.preprocessing import MinMaxScaler

#this script performes the domain adaptation experiment
#it tests the models created in cross-validation in a test set from another file

def KNN(train_file, sets, test_file1, test_file2):
    test = np.concatenate((np.load("External_test/Non_discretized/{}/Test.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = np.concatenate((np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    KNN = pickle.load(open('Cross_validation/Non-discretized/{}/Sets{}/KNN.sav'.format(train_file, sets), 'rb'))
    predict_labels = KNN.predict(test)
    predict_prob = KNN.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def RF(train_file, sets, test_file1, test_file2):
    test = np.concatenate((np.load("External_test/Non_discretized/{}/Test.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = np.concatenate((np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    RF = pickle.load(open('Cross_validation/Non-discretized/{}/Sets{}/RF.sav'.format(train_file, sets), 'rb'))
    predict_labels = RF.predict(test)
    predict_prob = RF.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc


def GaussianNaiveB(train_file, sets, test_file1, test_file2):
    test = np.concatenate((np.load("External_test/Non_discretized/{}/Test.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = np.concatenate((np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    NB = pickle.load(open('Cross_validation/Non-discretized/{}/Sets{}/NB.sav'.format(train_file, sets), 'rb'))
    predict_labels = NB.predict(test)
    predict_prob = NB.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def DT(train_file, sets, test_file1, test_file2):
    test = np.concatenate((np.load("External_test/Non_discretized/{}/Test.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = np.concatenate((np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    DT = pickle.load(open('Cross_validation/Non-discretized/{}/Sets{}/DT.sav'.format(train_file, sets), 'rb'))
    predict_labels = DT.predict(test)
    predict_prob = DT.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def Adaboost(train_file, sets, test_file1, test_file2):
    test = np.concatenate((np.load("External_test/Non_discretized/{}/Test.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = np.concatenate((np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    AD = pickle.load(open('Cross_validation/Non-discretized/{}/Sets{}/AD.sav'.format(train_file, sets), 'rb'))
    predict_labels = AD.predict(test)
    predict_prob = AD.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def svc(train_file, sets, test_file1, test_file2):
    test = np.concatenate((np.load("External_test/Non_discretized/{}/Test.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = np.concatenate((np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    transformer = Normalizer().fit(test)
    test = transformer.transform(test)

    svc = pickle.load(open('Cross_validation/Non-discretized/{}/Sets{}/svc.sav'.format(train_file, sets), 'rb'))
    predict_labels = svc.predict(test)
    predict_prob = svc.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def mlp(train_file, sets, test_file1, test_file2):
    test = np.concatenate((np.load("External_test/Non_discretized/{}/Test.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = np.concatenate((np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True), np.load("External_test/Non_discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    transformer = Normalizer().fit(test)
    test = transformer.transform(test)

    MLP = pickle.load(open('Cross_validation/Non-discretized/{}/Sets{}/svc.sav'.format(train_file, sets), 'rb'))
    predict_labels = MLP.predict(test)
    predict_prob = MLP.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def EFC_test_data(train_file, sets, test_file1, test_file2):
    test1 = np.load("External_test/Discretized/{}/Test.npy".format(test_file1), allow_pickle=True)
    test2 = np.load("External_test/Discretized/{}/Test.npy".format(test_file2), allow_pickle=True)
    test_normals = np.concatenate((test1[:int(len(test1)/2),:], test2[:int(len(test2)/2),:]), axis=0)
    test_malicious = np.concatenate((test1[int(len(test1)/2):,:], test2[int(len(test2)/2):,:]), axis=0)
    test_data = np.concatenate((test_normals, test_malicious), axis=0)
    test_labels1 = np.load("External_test/Discretized/{}/Test_labels.npy".format(test_file1), allow_pickle=True)
    test_labels2 = np.load("External_test/Discretized/{}/Test_labels.npy".format(test_file2), allow_pickle=True)
    test_labels_normals = np.concatenate((test_labels1[:int(len(test_labels1)/2)], test_labels2[:int(len(test_labels2)/2)]), axis=0)
    test_labels_malicious = np.concatenate((test_labels1[int(len(test_labels1)/2):], test_labels2[int(len(test_labels2)/2):]), axis=0)
    test_labels = np.concatenate((test_labels_normals, test_labels_malicious), axis=0)
    test_labels = [0 if x==1 else 1 for x in test_labels]
    Q = 14
    LAMBDA = 0.5


    model_normal = np.load("Cross_validation/Discretized/{}/Sets{}/model_normal.npy".format(train_file, sets), allow_pickle=True)
    h_i = np.load("Cross_validation/Discretized/{}/Sets{}/h_i.npy".format(train_file, sets), allow_pickle=True)
    CUTOFF = np.load("Cross_validation/Discretized/{}/Sets{}/cutoff.npy".format(train_file, sets), allow_pickle=True)

    predicted_labels, energies = test_model(test_data, model_normal, h_i, test_labels, CUTOFF, Q)
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies).reshape(-1,1))]

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    roc = roc_auc_score(test_labels, predict_prob)

    np.save("External_test/Results/Training_{}/Sets{}/results.npy".format(train_file, sets), np.array([precision, recall, f1, roc]))
    np.save("External_test/Results/Training_{}/Sets{}/energies.npy".format(train_file, sets), np.array(energies))
    return precision, recall, f1, roc

for sets in range(1,11):
    files = ['TFTP_01-12','Syn_03-11','DrDoS_NTP_01-12']
    for file in files:
        train_file = file
        complement = files.copy()
        complement.remove(file)
        test_file1 = files[0]
        test_file2 = files[1]

        EFC_list = []
        KNN_list = []
        RF_list = []
        GaussianNB_list = []
        DT_list = []
        Adaboost_list = []
        svc_list = []
        mlp_list = []


        precision, recall, f1, roc = EFC_test_data(train_file, sets, test_file1, test_file2)
        EFC_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = mlp(train_file, sets, test_file1, test_file2)
        mlp_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = KNN(train_file, sets, test_file1, test_file2)
        KNN_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = RF(train_file, sets, test_file1, test_file2)
        RF_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = GaussianNaiveB(train_file, sets, test_file1, test_file2)
        GaussianNB_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = DT(train_file, sets, test_file1, test_file2)
        DT_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = Adaboost(train_file, sets, test_file1, test_file2)
        Adaboost_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = svc(train_file, sets, test_file1, test_file2)
        svc_list = [precision, recall, f1, roc]

        np.save("External_test/Results/Training_{}/Sets{}/EFC.npy".format(train_file, sets), np.array(EFC_list))
        np.save("External_test/Results/Training_{}/Sets{}/ANN.npy".format(file, sets), np.array(mlp_list))
        np.save("External_test/Results/Training_{}/Sets{}/KNN.npy".format(train_file, sets), np.array(KNN_list))
        np.save("External_test/Results/Training_{}/Sets{}/RF.npy".format(train_file, sets), np.array(RF_list))
        np.save("External_test/Results/Training_{}/Sets{}/GaussianNB.npy".format(train_file, sets), np.array(GaussianNB_list))
        np.save("External_test/Results/Training_{}/Sets{}/DT.npy".format(train_file, sets), np.array(DT_list))
        np.save("External_test/Results/Training_{}/Sets{}/Adaboost.npy".format(train_file, sets), np.array(Adaboost_list))
        np.save("External_test/Results/Training_{}/Sets{}/SVC.npy".format(train_file, sets), np.array(svc_list))
