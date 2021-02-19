import numpy as np
import pandas as pd
from statistics import mean, stdev
from math import sqrt

algorithms = ['GaussianNB','KNN', 'DT', 'Adaboost', 'RF', 'SVC','MLP','EFC']
results = open("Data/Results/Mean_external_internal_results.txt", 'w+')
for j in algorithms:
    precision_internal = []
    recall_internal = []
    f1_internal = []
    roc_internal = []
    precision_external = []
    recall_external = []
    f1_external = []
    roc_external = []
    for i in range(1,11):
        metrics = np.load("Data/Results/Exp{}/{}_internal.npy".format(i, j))
        precision_internal.append(metrics[0])
        recall_internal.append(metrics[1])
        f1_internal.append(metrics[2])
        roc_internal.append(metrics[3])

        metrics = np.load("Data/Results/Exp{}/{}_external.npy".format(i, j))
        precision_external.append(metrics[0])
        recall_external.append(metrics[1])
        f1_external.append(metrics[2])
        roc_external.append(metrics[3])

    results.write('{} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} \\\\ \n'.format(j,
    mean(f1_internal), (stdev(f1_internal)/sqrt(len(f1_internal)))*1.96, mean(roc_internal), (stdev(roc_internal)/sqrt(len(f1_internal)))*1.96, mean(f1_external), (stdev(f1_external)/sqrt(len(f1_internal)))*1.96, mean(roc_external), (stdev(f1_internal)/sqrt(len(roc_external)))*1.96))
results.close()
