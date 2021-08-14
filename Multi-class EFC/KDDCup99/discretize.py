import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
from sklearn.model_selection import train_test_split
import os
import random
import sys

def get_intervals(file):
    bins = []
    for feature in range(41):
        data = pd.read_csv(file, usecols = [feature], header=None, squeeze=True)
        if is_numeric_dtype(data) and np.unique(data).shape[0] > 10:
            quantiles = np.quantile(data, [i*(1/50) for i in range(1, 51)])
            bins.append(list(np.unique(quantiles)))
        else:
            bins.append(list(np.unique(data)))
    return bins


def discretize(data, dict):
    for feature in range(41):
        col_values = data.iloc[:, feature]
        if is_numeric_dtype(col_values) and np.unique(col_values).shape[0] > 10:
            l_edge = np.NINF
            for x, r_edge in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in col_values]
                l_edge = r_edge
            data.iloc[:, feature] = [x if value > r_edge else value for value in col_values]
        else:
            diff = np.setdiff1d(col_values, dict[feature])
            if diff.shape[0] > 0:
                dict[feature] += [x for x in diff]
            data.iloc[:, feature] = [dict[feature].index(x) for x in col_values]
    print(np.unique(data))
    return data

malicious_names = [['normal.'], ['back.', 'smurf.', 'teardrop.', 'neptune.', 'land.', 'pod.'],
['ipsweep.',  'portsweep.',  'satan.',  'nmap.'], ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.',
 'warezclient.', 'warezmaster.', 'phf.'], ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']]


#load sets
train = pd.read_csv('Data/Normalized/train', header=None)
validation = pd.read_csv('Data/Normalized/validation', header=None)
test = pd.read_csv('Data/Normalized/corrected', header=None)

# encode labels
for idx in range(len(malicious_names)):
    train.iloc[:, -1] = [idx if value in malicious_names[idx] else value for value in train.iloc[:,-1]]
    validation.iloc[:, -1] = [idx if value in malicious_names[idx] else value for value in validation.iloc[:,-1]]
    test.iloc[:, -1] = [idx if value in malicious_names[idx] else value for value in test.iloc[:,-1]]
test.iloc[:, -1] = [100 if value not in range(len(malicious_names)) else value for value in test.iloc[:,-1]]


# remove dupkicates in unknown class to experiment 2
unknown_idx = np.where(test.iloc[:, -1]==100)[0]
X_unknown = test.iloc[unknown_idx, :]
new_test = test.drop(unknown_idx, axis=0, inplace=False)
X_unknown.drop_duplicates(subset=X_unknown.columns[1:-1], inplace=True)
test_unknown_unique = pd.concat([new_test, X_unknown], ignore_index=True)


# get intervals from training set
intervals = get_intervals('Data/Normalized/train')
np.save("Dict.npy", intervals)

# discretize train, validation, test and test_unknown_unique sets
train = discretize(train, intervals)
test = discretize(test, intervals)
test_unknown_unique = discretize(test_unknown_unique, intervals)
validation = discretize(validation, intervals)

# save
train.iloc[:, :-1].to_csv('Data/Normalized-Discretized/X_train', header=False, index=False)
train.iloc[:, -1].to_csv('Data/Normalized-Discretized/y_train', header=False, index=False)

test.iloc[:, :-1].to_csv('Data/Normalized-Discretized/X_test', header=False, index=False)
test.iloc[:, -1].to_csv('Data/Normalized-Discretized/y_test', header=False, index=False)

test_unknown_unique.iloc[:, :-1].to_csv('Data/Normalized-Discretized/X_test_unknown_unique', header=False, index=False)
test_unknown_unique.iloc[:, -1].to_csv('Data/Normalized-Discretized/y_test_unknown_unique', header=False, index=False)

validation.iloc[:, :-1].to_csv('Data/Normalized-Discretized/X_validation', header=False, index=False)
validation.iloc[:, -1].to_csv('Data/Normalized-Discretized/y_validation', header=False, index=False)
