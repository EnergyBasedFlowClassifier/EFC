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
from test_data_classic_samples import plot_energies, test_data_function
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import pickle
from sklearn.preprocessing import MinMaxScaler
from classification_functions import *

def KNN(sets):
    test = np.load("External_test/Non-discretized/test_ext_cidds-001.npy", allow_pickle=True)
    test_labels = np.load("External_test/Non-discretized/test_ext_labels_cidds-001.npy", allow_pickle=True)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    KNN = pickle.load(open('Cross_validation/Non-discretized/Sets{}/KNN.sav'.format(sets), 'rb'))
    predict_labels = KNN.predict(test)
    predict_prob = KNN.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def RF(sets):
    test = np.load("External_test/Non-discretized/test_ext_cidds-001.npy", allow_pickle=True)
    test_labels = np.load("External_test/Non-discretized/test_ext_labels_cidds-001.npy", allow_pickle=True)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    RF = pickle.load(open('Cross_validation/Non-discretized/Sets{}/RF.sav'.format(sets), 'rb'))
    predict_labels = RF.predict(test)
    predict_prob = RF.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc


def GaussianNaiveB(sets):
    test = np.load("External_test/Non-discretized/test_ext_cidds-001.npy", allow_pickle=True)
    test_labels = np.load("External_test/Non-discretized/test_ext_labels_cidds-001.npy", allow_pickle=True)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    NB = pickle.load(open('Cross_validation/Non-discretized/Sets{}/NB.sav'.format(sets), 'rb'))
    predict_labels = NB.predict(test)
    predict_prob = NB.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def DT(sets):
    test = np.load("External_test/Non-discretized/test_ext_cidds-001.npy", allow_pickle=True)
    test_labels = np.load("External_test/Non-discretized/test_ext_labels_cidds-001.npy", allow_pickle=True)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    DT = pickle.load(open('Cross_validation/Non-discretized/Sets{}/DT.sav'.format(sets), 'rb'))
    predict_labels = DT.predict(test)
    predict_prob = DT.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def Adaboost(sets):
    test = np.load("External_test/Non-discretized/test_ext_cidds-001.npy", allow_pickle=True)
    test_labels = np.load("External_test/Non-discretized/test_ext_labels_cidds-001.npy", allow_pickle=True)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    AD = pickle.load(open('Cross_validation/Non-discretized/Sets{}/AD.sav'.format(sets), 'rb'))
    predict_labels = AD.predict(test)
    predict_prob = AD.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def svc(sets):
    test = np.load("External_test/Non-discretized/test_ext_cidds-001.npy", allow_pickle=True)
    test_labels = np.load("External_test/Non-discretized/test_ext_labels_cidds-001.npy", allow_pickle=True)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    transformer = Normalizer().fit(test)
    test = transformer.transform(test)

    svc = pickle.load(open('Cross_validation/Non-discretized/Sets{}/svc.sav'.format(sets), 'rb'))
    predict_labels = svc.predict(test)
    predict_prob = svc.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def mlp(sets):
    test = np.load("External_test/Non-discretized/test_ext_cidds-001.npy", allow_pickle=True)
    test_labels = np.load("External_test/Non-discretized/test_ext_labels_cidds-001.npy", allow_pickle=True)
    test_labels = [0 if x==1 else 1 for x in test_labels]

    transformer = Normalizer().fit(test)
    test = transformer.transform(test)

    MLP = pickle.load(open('Cross_validation/Non-discretized/Sets{}/svc.sav'.format(sets), 'rb'))
    predict_labels = MLP.predict(test)
    predict_prob = MLP.predict_proba(test)
    precision = precision_score(test_labels, predict_labels)
    recall = recall_score(test_labels, predict_labels)
    f1 = f1_score(test_labels, predict_labels)
    roc = roc_auc_score(test_labels, predict_prob[:,1])
    return precision, recall, f1, roc

def EFC_test_data(sets):
    test_data = np.load("External_test/Discretized/test_ext_cidds-001.npy", allow_pickle=True)
    test_labels = np.load("External_test/Discretized/test_ext_labels_cidds-001.npy", allow_pickle=True)
    Q = 32
    LAMBDA = 0.5

    model_normal = np.load("Cross_validation/Discretized/Sets{}/model_normal.npy".format(sets), allow_pickle=True)
    h_i = np.load("Cross_validation/Discretized/Sets{}/h_i.npy".format(sets), allow_pickle=True)
    CUTOFF = np.load("Cross_validation/Discretized/Sets{}/cutoff.npy".format(sets), allow_pickle=True)

    predicted_labels, energies = test_model(test_data, model_normal, h_i, test_labels, CUTOFF, Q)

    predict_prob = [x for x in MinMaxScaler().fit_transform(np.array(energies).reshape(-1,1))]
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)
    f1 = f1_score(test_labels, predicted_labels)
    roc = roc_auc_score(test_labels, predict_prob)

    np.save("External_test/Results/Sets{}/results.npy".format(sets), np.array([precision, recall, f1, roc]))
    np.save("External_test/Results/Sets{}/energies.npy".format(sets), np.array(energies))
    return precision, recall, f1, roc


for sets in range(1,11):
    EFC_list = []
    KNN_list = []
    RF_list = []
    GaussianNB_list = []
    DT_list = []
    Adaboost_list = []
    svc_list = []
    mlp_list = []

    precision, recall, f1, roc = EFC_test_data(sets)
    EFC_list = [precision, recall, f1, roc]
    precision, recall, f1, roc = mlp(sets)
    mlp_list = [precision, recall, f1, roc]
    precision, recall, f1, roc = KNN(sets)
    KNN_list = [precision, recall, f1, roc]
    precision, recall, f1, roc = RF(sets)
    RF_list = [precision, recall, f1, roc]
    precision, recall, f1, roc = GaussianNaiveB(sets)
    GaussianNB_list = [precision, recall, f1, roc]
    precision, recall, f1, roc = DT(sets)
    DT_list = [precision, recall, f1, roc]
    precision, recall, f1, roc = Adaboost(sets)
    Adaboost_list = [precision, recall, f1, roc]
    precision, recall, f1, roc = svc(sets)
    svc_list = [precision, recall, f1, roc]

    np.save("External_test/Results/Sets{}/EFC.npy".format(sets), np.array(EFC_list))
    np.save("External_test/Results/Sets{}/ANN.npy".format(sets), np.array(mlp_list))
    np.save("External_test/Results/Sets{}/KNN.npy".format(sets), np.array(KNN_list))
    np.save("External_test/Results/Sets{}/RF.npy".format(sets), np.array(RF_list))
    np.save("External_test/Results/Sets{}/GaussianNB.npy".format(sets), np.array(GaussianNB_list))
    np.save("External_test/Results/Sets{}/DT.npy".format(sets), np.array(DT_list))
    np.save("External_test/Results/Sets{}/Adaboost.npy".format(sets), np.array(Adaboost_list))
    np.save("External_test/Results/Sets{}/SVC.npy".format(sets), np.array(svc_list))
