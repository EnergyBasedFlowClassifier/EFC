import numpy as np
import seaborn as sns
import pandas as pd
import os
import pickle
from statistics import mean, stdev
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, precision_recall_fscore_support
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt


def metrics_efc_multiclass():
    precision = [[],[],[],[],[]]
    recall = [[],[],[],[],[]]
    f1_score = [[],[],[],[],[]]
    support = [[],[],[],[],[]]

    for sets in range(1,6):
        y_true = np.load("5-fold_sets/Discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)
        y_pred = np.load("5-fold_sets/Results/Sets{}/EFC_predicted.npy".format(sets), allow_pickle=True)
        metrics = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4])

        for label in [0, 1, 2, 3, 4]:
            precision[label].append(metrics[0][label])
            recall[label].append(metrics[1][label])
            f1_score[label].append(metrics[2][label])
            support[label].append(metrics[3][label])

    names = ['normal','pingScan','bruteForce','portScan','dos']
    with open("5-fold_sets/Results/EFC_mean_folds.txt", 'w+') as file:
        for i, item in enumerate(names):
            file.write('\\textit{{{}}} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.0f}\\\\ \n'.format(item, mean(precision[i]),(stdev(precision[i])/sqrt(len(precision[i])))*1.96, mean(recall[i]), (stdev(recall[i])/sqrt(len(recall[i])))*1.96,mean(f1_score[i]),(stdev(f1_score[i])/sqrt(len(f1_score[i])))*1.96,mean(support[i])))
        file.close()

def metrics_algorithms_multiclass():
    with open("5-fold_sets/Results/Algorithms_mean_folds.txt", 'w+') as file:
        for alg in ['NB','KNN', 'DT', 'SVC', 'MLP','EFC','RF','AD']:
            precision = []
            recall = []
            f1_score = []
            for sets in range(1,6):
                y_true = np.load("5-fold_sets/Discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True)
                y_pred = np.load("5-fold_sets/Results/Sets{}/{}_predicted.npy".format(sets, alg), allow_pickle=True)
                metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted')
                precision.append(metrics[0])
                recall.append(metrics[1])
                f1_score.append(metrics[2])

            file.write('\\textit{{{}}} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f}\\\\ \n'.format(alg, mean(precision),(stdev(precision)/sqrt(len(precision)))*1.96, mean(recall), (stdev(recall)/sqrt(len(recall)))*1.96,mean(f1_score),(stdev(f1_score)/sqrt(len(f1_score)))*1.96))
        file.close()


def c_matrix_efc():
    labels = [0,1,2,3,4]
    y_true = np.load("5-fold_sets/Discretized/Sets1/test_labels.npy", allow_pickle=True)
    y_pred = np.load("5-fold_sets/Results/Sets1/EFC_predicted.npy", allow_pickle=True)

    names = ['normal','pingScan','bruteForce','portScan','dos']
    results = confusion_matrix(y_true, y_pred, labels=labels)
    conf = pd.DataFrame(results, index=names, columns=names)
    sns.set(font_scale=1.10)
    plt.figure(figsize=(9,9), dpi=100)
    fig = sns.heatmap(conf, annot=True, cmap="Blues",fmt='g', cbar=False)
    fig.set_xticklabels(names, rotation=40, ha='right')
    fig.set_yticklabels(names, rotation=40)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.savefig("5-fold_sets/Results/Confusion_matrix_CIDDS001.png", format="png")
    plt.show()

def unknown_matrix_efc():
    predicted = np.load("5-fold_sets/Results_removing{}/Sets{}/{}_predicted.npy".format(removed), allow_pickle=True)
    true = np.load("5-fold_sets/Discretized/Sets1/test_labels.npy", allow_pickle=True)

    names = ['normal','pingScan','bruteForce','portScan','dos']

    results = confusion_matrix(true, predicted, labels=np.unique(true))
    conf = pd.DataFrame(results, index=names, columns=names)
    sns.set(font_scale=1.5)
    plt.figure(figsize=(9,9), dpi=100)
    fig = sns.heatmap(conf, annot=True, cmap="Blues",fmt='g', cbar=False)
    fig.set_xticklabels(names, rotation=40, ha='right')
    fig.set_yticklabels(names, rotation=40)
    plt.subplots_adjust(left=0.15, bottom=0.15)
    plt.title("EFC - Leaving out dos")
    plt.savefig("5-fold_sets/Results/Dos_unknown_matrix_CIDDS001.png", format="png")
    plt.show()

def tables_unknown():
    names = ['normal','pingScan','bruteForce','portScan','dos']
    with open("5-fold_sets/Table_DT-EFC_unknown.txt", 'w+') as file:
        for removed in [1, 2, 3, 4]:
            for alg in ['DT','EFC']:
                normal_percent = []
                others_percent = []
                suspicious_percent = []
                for sets in range(1,6):
                    y_true = list(np.load("5-fold_sets/Discretized/Sets{}/test_labels.npy".format(sets), allow_pickle=True))
                    y_pred = list(np.load("5-fold_sets/Results_removing{}/Sets{}/{}_predicted.npy".format(removed, sets, alg), allow_pickle=True))
                    unknown_predicted = [y_pred[i] for i in np.where(np.array(y_true)==removed)[0]]
                    unique, counts = np.unique(unknown_predicted, return_counts=True)
                    print(unique)
                    print(counts)

                    normal_predicted = counts[np.where(unique==0)[0]]
                    if not normal_predicted:
                        normal_predicted = 0

                    suspicious_predicted = counts[np.where(unique==100)[0]]
                    if not suspicious_predicted:
                        suspicious_predicted = 0

                    print("normal", normal_predicted)
                    print("suspicious", suspicious_predicted)
                    normal_percent.append(float(normal_predicted/len(unknown_predicted)))
                    others_percent.append(float((len(unknown_predicted)-normal_predicted-suspicious_predicted)/len(unknown_predicted)))
                    suspicious_percent.append(float(suspicious_predicted/len(unknown_predicted)))

                print(others_percent)
                print(normal_percent)
                print(suspicious_percent)
                if alg != 'EFC':
                    file.write('\\textit{{{}}} & {:.2f} $\\pm$ {:.3f} & {:.2f} $\\pm$ {:.3f} & &'.format(names[removed], mean(normal_percent), (stdev(normal_percent)/sqrt(len(normal_percent)))*1.96, mean(others_percent), (stdev(others_percent)/sqrt(len(others_percent)))*1.96))
                else:
                    file.write('{:.2f} $\\pm$ {:.3f} & {:.2f} $\\pm$ {:.3f} & {:.2f} $\\pm$ {:.3f} \\\\ \n'.format(mean(normal_percent), (stdev(normal_percent)/sqrt(len(normal_percent)))*1.96, mean(others_percent), (stdev(others_percent)/sqrt(len(others_percent)))*1.96, mean(suspicious_percent), (stdev(suspicious_percent)/sqrt(len(suspicious_percent)))*1.96))
    file.close()

metrics_efc_multiclass()
# c_matrix_efc()
metrics_algorithms_multiclass()
tables_unknown()
# unknown_matrix_efc()
