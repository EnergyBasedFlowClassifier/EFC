import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
from sklearn.model_selection import train_test_split
import os
import random
import sys

train_origin = "Encoded/Normalized"
train_dest = "Encoded/Normalized-Discretized"

if sys.argv[1] == '0':
    test_origin = "Encoded/Normalized"
    test_dest = "Encoded/Normalized-Discretized"
if sys.argv[1] == '1':
    test_origin = "Encoded/Unique-Unknown"
    test_dest = "Encoded/Unique-Unknown-Discretized"


def get_intervals(file):
    intervals = []
    for feature in range(41):
        data = pd.read_csv(file, usecols = [feature], header=None, squeeze=True)
        if is_numeric_dtype(data) and len(np.unique(data)) > 10:
            _, retbins = pd.qcut(data, 150, labels=False, retbins=True, duplicates = 'drop')
            intervals.append(retbins)
        else: #symbolic, bool or small range
            intervals.append(list(np.unique(data)))
        print(intervals[feature], len(intervals[feature]))
    return intervals


def discretize(data, dict):
    for feature in range(41):
        col_val = data.iloc[:,feature]
        if is_numeric_dtype(col_val):
            l_edge = np.NINF
            for x, r_edge in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in data.iloc[:,feature]]
                if r_edge == dict[feature][-1]:
                    data.iloc[:, feature] = [x if value > r_edge else value for value in data.iloc[:,feature]]
                l_edge = r_edge

        else: #symbolic
            diff = np.setdiff1d(col_val, dict[feature])
            if diff.shape[0] > 0:
                dict[feature] += [x for x in diff]
            data.iloc[:,feature] = [dict[feature].index(x) for x in col_val]
        print(np.unique(data.iloc[:, feature]))
    return data

malicious_names = [['normal.'], ['back.', 'smurf.', 'teardrop.', 'neptune.', 'land.', 'pod.'],
['ipsweep.',  'portsweep.',  'satan.',  'nmap.'], ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.',
 'warezclient.', 'warezmaster.', 'phf.'], ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']]


# # discretize train, validation and test sets
# intervals = get_intervals('Data/{}/train'.format(train_origin))
# np.save("Dict.npy", intervals)

intervals = np.load("Dict.npy", allow_pickle=True)

data = pd.read_csv('Data/{}/train'.format(train_origin), header=None)
data = discretize(data, intervals)
data.iloc[:, :-1].to_csv('Data/{}/X_train'.format(train_dest), header=False, index=False)
data.iloc[:, -1].to_csv('Data/{}/y_train'.format(train_dest), header=False, index=False)

data = pd.read_csv('Data/{}/validation'.format(test_origin), header=None)
data = discretize(data, intervals)
data.iloc[:, :-1].to_csv('Data/{}/X_validation'.format(test_dest), header=False, index=False)
data.iloc[:, -1].to_csv('Data/{}/y_validation'.format(test_dest), header=False, index=False)

data = pd.read_csv('Data/{}/corrected'.format(test_origin), header=None)
data = discretize(data, intervals)
data.iloc[:, :-1].to_csv('Data/{}/X_test'.format(test_dest), header=False, index=False)
data.iloc[:, -1].to_csv('Data/{}/y_test'.format(test_dest), header=False, index=False)
