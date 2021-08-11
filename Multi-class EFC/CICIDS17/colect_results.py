import numpy as np
import seaborn as sns
import pandas as pd
import os
import pickle
from statistics import mean, stdev
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, precision_recall_fscore_support, f1_score, recall_score
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from scipy.stats import gmean, gstd

def metrics_algorithms_multiclass():
    names = ['BENIGN',  'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack',
    'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']
    with open("5-fold_sets/Results/Algorithms_comparison.txt", 'w+') as file:
        macro_avarege = [[],[],[],[],[],[],[],[]]
        balanced_acc = [[],[],[],[],[],[],[],[]]
        f1_scores = [[],[],[],[],[],[],[],[]]
        f1_scores_std = [[],[],[],[],[],[],[],[]]
        for idx, alg in enumerate(['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']):
            f1 = []
            for sets in range(1,6):
                y_true = pd.read_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(sets), header=None)
                y_pred = np.load("5-fold_sets/Results/Sets{}/{}_predicted.npy".format(sets, alg), allow_pickle=True)
                macro_avarege[idx].append(f1_score(y_true, y_pred, average='macro'))
                balanced_acc[idx].append(balanced_accuracy_score(y_true, y_pred))
                f1.append(f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))

            for label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                lista = []
                for x in range(5):
                    lista.append(f1[x][label])
                f1_scores[idx].append(mean(lista))
                f1_scores_std[idx].append(stdev(lista)/sqrt(len(lista))*1.96)

        print(macro_avarege)
        for label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            file.write('\\textit{{{}}} '.format(names[label]))
            for idx, alg in enumerate(['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']):
                file.write('& {:.3f} $\\pm$ {:.3f}'.format(f1_scores[idx][label], f1_scores_std[idx][label]))
            file.write('\\\\ \n')

        file.write('\\textbf{{{}}} '.format('Macro average'))
        for idx, alg in enumerate(['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']):
                file.write('& {:.3f} $\\pm$ {:.3f}'.format(mean(macro_avarege[idx]), (stdev(macro_avarege[idx])/sqrt(len(macro_avarege[idx])))*1.96))
        file.write('\\\\ \n')

        file.write('\\textbf{{{}}} '.format('Balanced accuracy'))
        for idx, alg in enumerate(['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']):
                file.write('& {:.3f} $\\pm$ {:.3f}'.format(mean(balanced_acc[idx]), (stdev(balanced_acc[idx])/sqrt(len(balanced_acc[idx])))*1.96))
        file.write('\\\\ \n')

def plot_unknown():
    names = ['DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack',
    'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(1,2, figsize=(16.8, 4.8))
    plt.tight_layout(w_pad=3.3)
    plt.ylim((0.0,1.4))
    width = 0.45
    for alg in ['RF','EFC']:
        removed_metrics = []
        for removed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            normal_percent = []
            others_percent = []
            suspicious_percent = []
            for sets in range(1,6):
                y_true = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(sets), header=None, squeeze=True))
                y_pred = list(np.load("5-fold_sets/Results_removing{}/Sets{}/{}_predicted.npy".format(removed, sets, alg), allow_pickle=True))
                unknown_predicted = [y_pred[i] for i in np.where(y_true==removed)[0]]
                print(len(unknown_predicted))
                unique, counts = np.unique(unknown_predicted, return_counts=True)

                normal_predicted = counts[np.where(unique==0)[0]]
                if not normal_predicted:
                    normal_predicted = 0

                suspicious_predicted = counts[np.where(unique==100)[0]]
                if not suspicious_predicted:
                    suspicious_predicted = 0

                normal_percent.append(float(normal_predicted/len(unknown_predicted)))
                others_percent.append(float((len(unknown_predicted)-normal_predicted-suspicious_predicted)/len(unknown_predicted)))
                suspicious_percent.append(float(suspicious_predicted/len(unknown_predicted)))

            if alg != 'EFC':
                removed_metrics.append([mean(normal_percent), (stdev(normal_percent)/sqrt(len(normal_percent)))*1.96, mean(others_percent), (stdev(others_percent)/sqrt(len(others_percent)))*1.96])
            else:
                removed_metrics.append([mean(normal_percent), (stdev(normal_percent)/sqrt(len(normal_percent)))*1.96, mean(others_percent), (stdev(others_percent)/sqrt(len(others_percent)))*1.96, mean(suspicious_percent), (stdev(suspicious_percent)/sqrt(len(suspicious_percent)))*1.96])

        if alg != 'EFC':
            ax[0].bar(names, [x[0] for x in removed_metrics], width, capsize=3, ecolor='black', yerr=[x[1] for x in removed_metrics], label='Benign', color='#006BA4')
            ax[0].bar(names, [x[2] for x in removed_metrics], width, capsize=3, ecolor='black', yerr=[x[3] for x in removed_metrics], bottom=[x[0] for x in removed_metrics], label='Other classes', color = '#CFCFCF')
            ax[0].set_ylabel('Percentages of predicted samples')
            ticks_loc = ax[0].get_xticks()
            ax[0].set_xticks(ax[0].get_xticks())
            ax[0].set_xticklabels(names, rotation=45, ha='right')
            ax[0].set_ylim((0.0,1.4))
            ax[0].legend(loc=1, bbox_to_anchor=(1.1, 1.05))
            ax[0].set_title("RF")
        else:
            ax[1].bar(names, [x[0] for x in removed_metrics], width, capsize=3, ecolor='black', yerr=[x[1] for x in removed_metrics], label='Benign', color='#006BA4')
            ax[1].bar(names, [x[2] for x in removed_metrics], width, capsize=3, ecolor='black', yerr=[x[3] for x in removed_metrics], bottom=[x[0] for x in removed_metrics], label='Other classes',  color = '#CFCFCF')
            ax[1].bar(names, [x[4] for x in removed_metrics], width, capsize=3, ecolor='black', yerr=[x[5] for x in removed_metrics], bottom=[x[0]+x[2] for x in removed_metrics], label='Suspicious',  color = '#FF800E')
            ax[1].set_ylabel('Percentages of predicted samples')
            ax[1].set_xticklabels(names, rotation=45, ha='right')
            ax[1].set_ylim((0.0,1.4))
            ax[1].legend(loc=1, bbox_to_anchor=(1.1, 1.05))
            ax[1].set_title("EFC")
        fig.savefig("5-fold_sets/Results/EFC_RF_unknown_CICIDS17.pdf", format="pdf",bbox_inches = "tight")#

def times():
    for alg in ['RF','NB','KNN', 'SVC', 'MLP', 'AD', 'DT','EFC']:
        train = []
        test = []
        for i in range(1,6):
            times = np.load("5-fold_sets/Results/Sets{}/{}_times.npy".format(i, alg), allow_pickle=True)
            train.append(times[0])
            test.append(times[1])
        print("{} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} \\\\".format(alg, mean(train), 1.96*stdev(train)/sqrt(len(train)), mean(test), 1.96*stdev(test)/sqrt(len(test))))

#metrics_algorithms_multiclass()
# times()
plot_unknown()
