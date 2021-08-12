import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import os

def get_intervals(file, n_bins):
    intervals = []
    for feature in range(8):
        data = pd.read_csv(file, usecols = [feature], header=None, squeeze=True)
        if is_numeric_dtype(data) and len(np.unique(data)) > 10:
            _, retbins = pd.qcut(data, n_bins, labels=False, retbins=True, duplicates = 'drop')
            intervals.append(retbins.astype('float64'))
        else:
            intervals.append(list(np.unique(data)))

    return intervals

def discretize(data, intervals):
    for feature in range(8):
        col_values = data.iloc[:,feature]
        if is_numeric_dtype(col_values) and len(np.unique(col_values)) > 10:
            data.iloc[:,feature] = pd.cut(col_values, intervals[feature], labels=False, include_lowest=True, duplicates = 'drop')
            data.iloc[:,feature].fillna(len(intervals[feature]), inplace=True)
        else:
            diff = np.setdiff1d(col_values, intervals[feature])
            if diff.shape[0] > 0:
                intervals[feature] += [x for x in diff]
            data.iloc[:,feature] = [intervals[feature].index(x) for x in col_values]
    print(np.unique(data))
    return data.astype('int')


malicious_names = ['normal','pingScan','bruteForce','portScan','dos']

for fold in range(1,6):
    intervals = get_intervals("5-fold_sets/Normalized/Sets{}/reduced_train.csv".format(fold), 30)
    np.save("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), intervals)

    # intervals = np.load("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), allow_pickle=True)
    reader = pd.read_csv("5-fold_sets/Normalized/Sets{}/reduced_train.csv".format(fold), chunksize=7000000, header=None)
    for chunk in reader:
        data = discretize(chunk, intervals)
        data.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(fold), mode='a', header=False, index=False)

    reader = pd.read_csv("5-fold_sets/Normalized/Sets{}/test.csv".format(fold), chunksize=7000000, header=None)
    for chunk in reader:
        data = discretize(chunk, intervals)
        data.to_csv("5-fold_sets/Discretized/Sets{}/test.csv".format(fold), mode='a', header=False, index=False)

    train_labels =  pd.read_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(fold), header=None)
    test_labels =  pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(fold), header=None)
    for i, value in enumerate(malicious_names):
        train_labels.iloc[:,-1][train_labels.iloc[:,-1] == value] = i
        test_labels.iloc[:,-1][test_labels.iloc[:,-1] == value] = i
    train_labels.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train_labels.csv".format(fold), header=False, index=False)
    test_labels.to_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(fold), header=False, index=False)
