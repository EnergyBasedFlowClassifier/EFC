import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype,  is_string_dtype
from matplotlib import pyplot as plt
import math


def get_intervals(file, n_bins):
    intervals = []
    for feature in range(79):
        data = pd.read_csv(file, usecols = [feature], header=None, squeeze=True)
        if is_numeric_dtype(data) and np.unique(data).shape[0] > 10:
            quantiles = np.quantile(data, [i*(1/n_bins) for i in range(1, n_bins+1)])
            intervals.append(list(np.unique(quantiles)))
        else:
            intervals.append(list(np.unique(data)))

    return intervals

def discretize(data, intervals):
    for feature in range(79):
        col_values = data.iloc[:,feature]
        bins = dict[feature]
        if is_numeric_dtype(col_values) and np.unique(col_values).shape[0] > 10:
            l_edge = np.NINF
            for x, r_edge in enumerate(bins):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in col_values]
                l_edge = r_edge
            data.iloc[:, feature] = [x if value > r_edge else value for value in col_values]
        else:
            diff = np.setdiff1d(col_values, bins)
            if diff.shape[0] > 0:
                bins += [x for x in diff]
            data.iloc[:, feature] = [bins.index(x) for x in col_values]
    print(np.unique(data))
    return data.astype('int')


malicious_names = ['BENIGN',  'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack',
'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']

for fold in range(1,6):
    intervals = get_intervals("5-fold_sets/Normalized/Sets{}/reduced_train.csv".format(fold), 30)
    np.save("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), intervals)

    # intervals = np.load("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), allow_pickle=True)
    reader = pd.read_csv("5-fold_sets/Normalized/Sets{}/reduced_train.csv".format(fold), chunksize=2_000_000, header=None)
    for chunk in reader:
        data = discretize(chunk, intervals)
        data.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(fold), mode='a', header=False, index=False)

    reader = pd.read_csv("5-fold_sets/Normalized/Sets{}/test.csv".format(fold), chunksize=2_000_000, header=None)
    for chunk in reader:
        data = discretize(chunk, intervals)
        data.to_csv("5-fold_sets/Discretized/Sets{}/test.csv".format(fold), mode='a', header=False, index=False)

    train_labels =  pd.read_csv("5-fold_sets/Normalized/Sets{}/reduced_train_labels.csv".format(fold), header=None)
    test_labels =  pd.read_csv("5-fold_sets/Normalized/Sets{}/test_labels.csv".format(fold), header=None)
    for i, value in enumerate(malicious_names):
        train_labels.iloc[:,-1][train_labels.iloc[:,-1] == value] = i
        test_labels.iloc[:,-1][test_labels.iloc[:,-1] == value] = i
    train_labels.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train_labels.csv".format(fold), header=False, index=False)
    test_labels.to_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(fold), header=False, index=False)
