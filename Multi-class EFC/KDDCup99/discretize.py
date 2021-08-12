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

def get_intervals(file, n_bins, columns):
    intervals = []
    for feature, name in enumerate(columns):
        data = pd.read_csv(file, usecols = [feature], header=None, squeeze=True)
        if name in ['protocol_type', 'service', 'flag','land','wrong_fragment',
                    'logged_in','root_shell', 'su_attempted','is_host_login', 'is_guest_login']: #symbolic or discrete
            intervals.append(list(np.unique(data)))
        else:
            _, retbins = pd.qcut(data, n_bins, labels=False, retbins=True, duplicates = 'drop')
            np.nan_to_num(retbins, copy=False, nan=0.0, neginf=0.0)
            intervals.append(np.unique(retbins).astype('float64'))
    return intervals

def discretize(data, intervals, columns):
    for feature, name in enumerate(columns):
        col_values = data.iloc[:,feature]
        if name in ['protocol_type', 'service', 'flag','land','wrong_fragment',
                    'logged_in','root_shell', 'su_attempted','is_host_login', 'is_guest_login']: #symbolic or discrete
            diff = np.setdiff1d(col_values, intervals[feature])
            if diff.shape[0] > 0:
                intervals[feature] += [x for x in diff]
            data.iloc[:,feature] = [intervals[feature].index(x) for x in col_values]
        else:
            data.iloc[:,feature] = pd.cut(col_values, intervals[feature], labels=False, include_lowest=True, duplicates = 'drop')
            data.iloc[:,feature].fillna(len(intervals[feature]), inplace=True)
        print(np.unique(data.iloc[:,feature]))

    return data.astype('int')

columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']


malicious_names = [['normal.'], ['back.', 'smurf.', 'teardrop.', 'neptune.', 'land.', 'pod.'],
['ipsweep.',  'portsweep.',  'satan.',  'nmap.'], ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.',
 'warezclient.', 'warezmaster.', 'phf.'], ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']]


# discretize train, validation and test sets
intervals = get_intervals('Data/{}/train'.format(train_origin), 200, columns)
np.save("Dict.npy", intervals)
intervals = np.load("Dict.npy", allow_pickle=True)

data = pd.read_csv('Data/{}/train'.format(train_origin), header=None)
data = discretize(data, intervals, columns)
data.iloc[:, :-1].to_csv('Data/{}/X_train'.format(train_dest), header=False, index=False)
data.iloc[:, -1].to_csv('Data/{}/y_train'.format(train_dest), header=False, index=False)

data = pd.read_csv('Data/{}/validation'.format(test_origin), header=None)
data = discretize(data, intervals, columns)
data.iloc[:, :-1].to_csv('Data/{}/X_validation'.format(test_dest), header=False, index=False)
data.iloc[:, -1].to_csv('Data/{}/y_validation'.format(test_dest), header=False, index=False)

data = pd.read_csv('Data/{}/corrected'.format(test_origin), header=None)
data = discretize(data, intervals, columns)
data.iloc[:, :-1].to_csv('Data/{}/X_test'.format(test_dest), header=False, index=False)
data.iloc[:, -1].to_csv('Data/{}/y_test'.format(test_dest), header=False, index=False)
