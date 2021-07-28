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
    names = ['normal','pingScan','bruteForce','portScan','dos']
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
                f1.append(f1_score(y_true, y_pred, average=None, labels=[0,1,2,3,4]))

            for label in [0,1,2,3,4]:
                lista = []
                for x in range(5):
                    lista.append(f1[x][label])
                f1_scores[idx].append(mean(lista))
                f1_scores_std[idx].append(stdev(lista)/sqrt(len(lista))*1.96)

        print(macro_avarege)
        for label in [0,1,2,3,4]:
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

def is_diff(removed_metrics):
    #0 - benign mean, 1- benign std, 2- other classes mean, 3- other classes std, 4 - benign mean, 5- benign std
    DT_benign_interval = [removed_metrics[0]-removed_metrics[1], removed_metrics[0]+removed_metrics[1]]
    EFC_benign_interval = [removed_metrics[4]-removed_metrics[5], removed_metrics[4]+removed_metrics[5]]
    if ((DT_benign_interval[0] <= EFC_benign_interval[1]) & (EFC_benign_interval[0] <= DT_benign_interval[1])):
        return "No"
    else: return "Yes"

def tables_unknown():
    names = ['normal','pingScan','bruteForce','portScan','dos']
    with open("5-fold_sets/Table_DT-EFC_unknown.txt", 'w+') as file:
        for removed in [1, 2, 3, 4]:
            removed_metrics = []
            for alg in ['RF','EFC']:
                normal_percent = []
                others_percent = []
                suspicious_percent = []
                for sets in range(1,6):
                    y_true = list(pd.read_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(sets), header=None))
                    y_pred = list(np.load("5-fold_sets/Results_removing{}/Sets{}/{}_predicted.npy".format(removed, sets, alg), allow_pickle=True))
                    unknown_predicted = [y_pred[i] for i in np.where(np.array(y_true)==removed)[0]]
                    normal_predicted = counts[np.where(unique==0)[0]]
                    if not normal_predicted:
                        normal_predicted = 0

                    suspicious_predicted = counts[np.where(unique==100)[0]]
                    if not suspicious_predicted:
                        suspicious_predicted = 0

                    normal_percent.append(float(normal_predicted/len(unknown_predicted)))
                    others_percent.append(float((len(unknown_predicted)-normal_predicted-suspicious_predicted)/len(unknown_predicted)))
                    suspicious_percent.append(float(suspicious_predicted/len(unknown_predicted)))

                removed_metrics += [mean(normal_percent), (stdev(normal_percent)/sqrt(len(normal_percent)))*1.96, mean(others_percent), (stdev(others_percent)/sqrt(len(others_percent)))*1.96]

                if alg != 'EFC':
                    file.write('\\textit{{\\textbf{{{}}}}} & \\textbf{{{:.2f}}} $\\pm$ \\textbf{{{:.3f}}} & {:.2f} $\\pm$ {:.3f} & &'.format(names[removed], mean(normal_percent), (stdev(normal_percent)/sqrt(len(normal_percent)))*1.96, mean(others_percent), (stdev(others_percent)/sqrt(len(others_percent)))*1.96))
                else:
                    file.write('\\textbf{{{:.2f}}} $\\pm$ \\textbf{{{:.3f}}} & {:.2f} $\\pm$ {:.3f} & {:.2f} $\\pm$ {:.3f} &'.format(mean(normal_percent), (stdev(normal_percent)/sqrt(len(normal_percent)))*1.96, mean(others_percent), (stdev(others_percent)/sqrt(len(others_percent)))*1.96, mean(suspicious_percent), (stdev(suspicious_percent)/sqrt(len(suspicious_percent)))*1.96))
            file.write('\\textbf{{{}}} \\\\ \n'.format(is_diff(removed_metrics)))
    file.close()

def plot_unknown():
    names = ['pingScan','bruteForce','portScan','DoS']
    plt.rcParams.update({'font.size': 16})

    fig, ax = plt.subplots(1,2, figsize=(16.8, 4.8))
    plt.tight_layout(w_pad=3.3)
    plt.ylim((0.0,1.4))
    width = 0.30
    for alg in ['RF','EFC']:
        removed_metrics = []
        for removed in [1, 2, 3, 4]:
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
            ax[0].bar(names, [x[0] for x in removed_metrics], width, yerr=[x[1] for x in removed_metrics], capsize=3, ecolor='black', label='Benign', color='#006BA4')
            ax[0].bar(names, [x[2] for x in removed_metrics], width, yerr=[x[3] for x in removed_metrics], capsize=3,  ecolor='black', bottom=[x[0] for x in removed_metrics], label='Other classes', color = '#CFCFCF')
            ax[0].set_ylabel('Percentages of predicted samples')
            ticks_loc = ax[0].get_xticks()
            ax[0].set_xticks(ax[0].get_xticks())
            ax[0].set_xticklabels(names, rotation=45, ha='right')
            ax[0].set_ylim((0.0,1.4))
            ax[0].legend(loc=1,bbox_to_anchor=(1.1, 1.05))
            ax[0].set_title("RF")
        else:
            ax[1].bar(names, [x[0] for x in removed_metrics], width, yerr=[x[1] for x in removed_metrics], capsize=3, ecolor='black', label='Benign', color='#006BA4')
            ax[1].bar(names, [x[2] for x in removed_metrics], width, yerr=[x[3] for x in removed_metrics], bottom=[x[0] for x in removed_metrics], capsize=3,  ecolor='black', label='Other classes',  color = '#CFCFCF')
            ax[1].bar(names, [x[4] for x in removed_metrics], width, yerr=[x[5] for x in removed_metrics], bottom=[x[0]+x[2] for x in removed_metrics], capsize=3,  ecolor='black', label='Suspicious',  color = '#FF800E')
            ax[1].set_ylabel('Percentages of predicted samples')
            ax[1].set_xticklabels(names, rotation=45, ha='right')
            ax[1].set_ylim((0.0,1.4))
            ax[1].legend(loc=1, bbox_to_anchor=(1.1, 1.05))
            ax[1].set_title("EFC")
        fig.savefig("5-fold_sets/Results/EFC_RF_unknown_CIDDS001.pdf", format="pdf",bbox_inches = "tight")#
        fig.savefig("5-fold_sets/Results/EFC_RF_unknown_CIDDS001.jpeg", format="pdf",bbox_inches = "tight")#

def times():
    for alg in ['RF','NB','KNN', 'SVC', 'MLP', 'AD', 'DT','EFC']:
        train = []
        test = []
        for i in range(1,6):
            times = np.load("5-fold_sets/Results/Sets{}/{}_times.npy".format(i, alg), allow_pickle=True)
            train.append(times[0])
            test.append(times[1])
        print("{} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} \\\\".format(alg, mean(train), 1.96*stdev(train)/sqrt(len(train)), mean(test), 1.96*stdev(test)/sqrt(len(test))))

# plot_unknown()
metrics_algorithms_multiclass()
# tables_unknown()
# unknown_matrix_efc()
# times()
