import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import os

def get_intervals(file, columns):
    intervals = []
    for feature in range(len(columns)):
        print(feature)
        data = pd.read_csv(file, usecols = [feature], header=None)
        data = list(data.iloc[:,0])
        if feature in [1,6]:
            intervals.append(list(np.unique(data)))
        else:
            if len(np.unique(data)) > 10:
                quantiles = np.quantile(data, [x*1/30 for x in range(1,31)])
                quantiles = sorted(list(set(quantiles)))
                intervals.append(quantiles)
            else:
                intervals.append(list(np.unique(data)))
        print(intervals[feature])
    return intervals


def discretize(data, dict):
    for feature in range(8):
        if feature in [1,6]:
            diff = np.setdiff1d(data.iloc[:, feature], dict[feature])
            if diff.shape[0] > 0:
                dict[feature] += [x for x in diff]
            for x, string in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value == string else value for value in data.iloc[:,feature]]

        else:
            l_edge = np.NINF
            for x, r_edge in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in data.iloc[:,feature]]
                if r_edge == dict[feature][-1]:
                    data.iloc[:, feature] = [x if value > r_edge else value for value in data.iloc[:,feature]]
                l_edge = r_edge
    return data

columns = ['Duration','Proto','Src Pt','Dst Pt','Packets','Bytes','Flags','Tos']
malicious_names = ['normal','pingScan','bruteForce','portScan','dos']


for fold in range(1,6):
    intervals = get_intervals("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), columns)
    np.save("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), intervals)

    # intervals = np.load("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), allow_pickle=True)
    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), chunksize=7000000, header=None)
    for chunk in reader:
        data = discretize(chunk, intervals)
        data.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(fold), mode='a', header=False, index=False)

    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test.csv".format(fold), chunksize=7000000, header=None)
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
