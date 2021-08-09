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
    test_origin = "Encoded/Unique-Unknown-Normalized"
    test_dest = "Encoded/Unique-Unknown-Normalized-Discretized"


def get_intervals(file):
    intervals = []
    for feature in range(41):
        print(feature)
        data = pd.read_csv(file, usecols = [feature], header=None)
        data = list(data.iloc[:,0])
        if feature in [1,2,3]:
            intervals.append(list(np.unique(data)))
        else:
            if len(np.unique(data)) > 10:
                quantiles = np.quantile(data, [i*(1/150) for i in range(1, 151)])
                quantiles = sorted(list(set(quantiles)))
                intervals.append(quantiles)
            else:
                intervals.append(list(np.unique(data)))
        print(intervals[feature], len(intervals[feature]))
    return intervals


def discretize(data, dict):
    for feature in range(41):
        if feature in [1,2,3]:
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
    print(np.unique(data))
    return data

malicious_names = [['normal.'], ['back.', 'smurf.', 'teardrop.', 'neptune.', 'land.', 'pod.'],
['ipsweep.',  'portsweep.',  'satan.',  'nmap.'], ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.',
 'warezclient.', 'warezmaster.', 'phf.'], ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']]


# discretize train, validation and test sets
intervals = get_intervals('Data/{}/train'.format(train_origin))
np.save("Dict.npy", intervals)
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
