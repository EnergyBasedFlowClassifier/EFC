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
from matplotlib import pyplot as plt
from cutoff import define_cutoff, plot_energies
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

#this script performes a cross validation for EFC and other ML classifiers

def KNN(file, sets):
    train = np.load("Cross_validation/Non-discretized/{}/Sets{}/train.npy".format(file, sets), allow_pickle=True)
    test = np.load("Cross_validation/Non-discretized/{}/Sets{}/test.npy".format(file, sets), allow_pickle=True)
    test_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/test_labels.npy".format(file, sets), allow_pickle=True)
    train_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/train_labels.npy".format(file, sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    KNN = KNeighborsClassifier()
    KNN.fit(train, train_labels)
    pickle.dump(KNN, open('Cross_validation/Non-discretized/{}/Sets{}/KNN.sav'.format(file, sets), 'wb'))
    predict_labels = KNN.predict(test)
    predict_prob = KNN.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def RF(file, sets):
    train = np.load("Cross_validation/Non-discretized/{}/Sets{}/train.npy".format(file, sets), allow_pickle=True)
    test = np.load("Cross_validation/Non-discretized/{}/Sets{}/test.npy".format(file, sets), allow_pickle=True)
    test_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/test_labels.npy".format(file, sets), allow_pickle=True)
    train_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/train_labels.npy".format(file, sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]


    RF = RandomForestClassifier()
    RF.fit(train, train_labels)
    pickle.dump(RF, open('Cross_validation/Non-discretized/{}/Sets{}/RF.sav'.format(file, sets), 'wb'))
    predict_labels = RF.predict(test)
    predict_prob = RF.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def GaussianNaiveB(file, sets):
    train = np.load("Cross_validation/Non-discretized/{}/Sets{}/train.npy".format(file, sets), allow_pickle=True)
    test = np.load("Cross_validation/Non-discretized/{}/Sets{}/test.npy".format(file, sets), allow_pickle=True)
    test_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/test_labels.npy".format(file, sets), allow_pickle=True)
    train_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/train_labels.npy".format(file, sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    NB = GaussianNB()
    NB.fit(train, train_labels)
    pickle.dump(NB, open('Cross_validation/Non-discretized/{}/Sets{}/NB.sav'.format(file, sets), 'wb'))
    predict_labels = NB.predict(test)
    predict_prob = NB.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def DT(file, sets):
    train = np.load("Cross_validation/Non-discretized/{}/Sets{}/train.npy".format(file, sets), allow_pickle=True)
    test = np.load("Cross_validation/Non-discretized/{}/Sets{}/test.npy".format(file, sets), allow_pickle=True)
    test_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/test_labels.npy".format(file, sets), allow_pickle=True)
    train_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/train_labels.npy".format(file, sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    DT = DecisionTreeClassifier()
    DT.fit(train, train_labels)
    pickle.dump(DT, open('Cross_validation/Non-discretized/{}/Sets{}/DT.sav'.format(file, sets), 'wb'))
    predict_labels = DT.predict(test)
    predict_prob = DT.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def Adaboost(file, sets):
    train = np.load("Cross_validation/Non-discretized/{}/Sets{}/train.npy".format(file, sets), allow_pickle=True)
    test = np.load("Cross_validation/Non-discretized/{}/Sets{}/test.npy".format(file, sets), allow_pickle=True)
    test_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/test_labels.npy".format(file, sets), allow_pickle=True)
    train_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/train_labels.npy".format(file, sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    AD = AdaBoostClassifier()
    AD.fit(train, train_labels)
    pickle.dump(AD, open('Cross_validation/Non-discretized/{}/Sets{}/AD.sav'.format(file, sets), 'wb'))
    predict_labels = AD.predict(test)
    predict_prob = AD.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def svc(file, sets):
    train = np.load("Cross_validation/Non-discretized/{}/Sets{}/train.npy".format(file, sets), allow_pickle=True)
    test = np.load("Cross_validation/Non-discretized/{}/Sets{}/test.npy".format(file, sets), allow_pickle=True)
    test_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/test_labels.npy".format(file, sets), allow_pickle=True)
    train_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/train_labels.npy".format(file, sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)

    svc = SVC(kernel='poly', probability=True)
    svc.fit(train, train_labels)
    pickle.dump(svc, open('Cross_validation/Non-discretized/{}/Sets{}/svc.sav'.format(file, sets), 'wb'))
    predict_labels = svc.predict(test)
    predict_prob = svc.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def mlp(file, sets):
    train = np.load("Cross_validation/Non-discretized/{}/Sets{}/train.npy".format(file, sets), allow_pickle=True)
    test = np.load("Cross_validation/Non-discretized/{}/Sets{}/test.npy".format(file, sets), allow_pickle=True)
    test_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/test_labels.npy".format(file, sets), allow_pickle=True)
    train_labels = np.load("Cross_validation/Non-discretized/{}/Sets{}/train_labels.npy".format(file, sets), allow_pickle=True)
    test_labels = [0 if x=='BENIGN' else 1 for x in test_labels]
    train_labels = [0 if x=='BENIGN' else 1 for x in train_labels]

    transformer = Normalizer().fit(train)
    train = transformer.transform(train)
    transformer = Normalizer().fit(test)
    test = transformer.transform(test)

    MLP = MLPClassifier(max_iter=300)
    MLP.fit(train, train_labels)
    pickle.dump(MLP, open('Cross_validation/Non-discretized/{}/Sets{}/MLP.sav'.format(file, sets), 'wb'))
    predict_labels = MLP.predict(test)
    predict_prob = MLP.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def EFC_test_data(file, sets):
    TEST_FILE = "Cross_validation/Discretized/{}/Sets{}/test.npy".format(file, sets)
    TRAIN_FILE = "Cross_validation/Discretized/{}/Sets{}/train.npy".format(file, sets)
    Q = 11
    LAMBDA = 0.5

    normal_data = list(np.load(TRAIN_FILE, allow_pickle=True))
    model_normal, h_i = create_model(np.array(normal_data,dtype=int), Q, LAMBDA)
    np.save("Cross_validation/Discretized/{}/Sets{}/h_i".format(file, sets), h_i)
    np.save("Cross_validation/Discretized/{}/Sets{}/model_normal".format(file, sets), model_normal)


    test_data = np.array(list(np.load(TEST_FILE, allow_pickle=True)))
    test_labels = np.array(list(np.load("Cross_validation/Discretized/{}/Sets{}/test_labels.npy".format(file, sets))))
    test_labels = [0 if x==1 else 1 for x in test_labels]
    CUTOFF = define_cutoff(TRAIN_FILE, file, sets)

    predicted_labels, energies = test_model(test_data, model_normal, h_i, test_labels, CUTOFF, Q)
    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies).reshape(-1,1))]

    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    roc = roc_auc_score(test_labels, predict_prob)

    np.save("Cross_validation/Discretized/{}/Sets{}/results.npy".format(file, sets), np.array([precision, recall, f1, roc]))
    np.save("Cross_validation/Discretized/{}/Sets{}/energies.npy".format(file, sets), np.array(energies))
    np.save("Cross_validation/Discretized/{}/Sets{}/cutoff.npy".format(file, sets), np.array(CUTOFF))
    return precision, recall, f1, roc

files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX','Wednesday-workingHours.pcap_ISCX']
for file in files:
    for sets in range(1,11):
        EFC_list = []
        KNN_list = []
        RF_list = []
        GaussianNB_list = []
        DT_list = []
        Adaboost_list = []
        svc_list = []
        mlp_list = []

        precision, recall, f1, roc = EFC_test_data(file, sets)
        EFC_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = mlp(file, sets)
        mlp_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = KNN(file, sets)
        KNN_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = RF(file, sets)
        RF_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = GaussianNaiveB(file, sets)
        GaussianNB_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = DT(file, sets)
        DT_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = Adaboost(file, sets)
        Adaboost_list = [precision, recall, f1, roc]
        precision, recall, f1, roc = svc(file, sets)
        svc_list = [precision, recall, f1, roc]


        np.save("Cross_validation/Results/{}/Sets{}/EFC.npy".format(file, sets), np.array(EFC_list))
        np.save("Cross_validation/Results/{}/Sets{}/ANN.npy".format(file, sets), np.array(mlp_list))
        np.save("Cross_validation/Results/{}/Sets{}/KNN.npy".format(file, sets), np.array(KNN_list))
        np.save("Cross_validation/Results/{}/Sets{}/RF.npy".format(file, sets), np.array(RF_list))
        np.save("Cross_validation/Results/{}/Sets{}/GaussianNB.npy".format(file, sets), np.array(GaussianNB_list))
        np.save("Cross_validation/Results/{}/Sets{}/DT.npy".format(file, sets), np.array(DT_list))
        np.save("Cross_validation/Results/{}/Sets{}/Adaboost.npy".format(file, sets), np.array(Adaboost_list))
        np.save("Cross_validation/Results/{}/Sets{}/SVC.npy".format(file, sets), np.array(svc_list))
