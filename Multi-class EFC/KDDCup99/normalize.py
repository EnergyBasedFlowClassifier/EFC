import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
from sklearn.model_selection import train_test_split
import os
import random
import sys

train_origin = "Encoded"
train_dest = "Encoded/Normalized"

if sys.argv[1] == '0':
    test_origin = "Encoded"
    test_dest = "Encoded/Normalized"
if sys.argv[1] == '1':
    test_origin = "Encoded/Unique-Unknown"
    test_dest = "Encoded/Unique-Unknown-Normalized"



def get_min_max(file):
    intervals = []
    for feature in range(41):
        data = pd.read_csv(file, usecols = [feature], header=None, squeeze=True)
        intervals.append([min(data), max(data)])
    return intervals

def normalize(data, dict, columns):
    for feature, name in enumerate(columns):
        if name in ['protocol_type', 'service', 'flag']: #symbolic
            pass
        elif name in ['land','wrong_fragment','logged_in','root_shell', 'su_attempted','is_host_login', 'is_guest_login']: #discrete
            pass
        # elif name in ['src_bytes', 'dst_bytes']: #log scalling
        #     data.iloc[:, feature] = np.log(data.iloc[:, feature])
        else:   #min max scalling
            print(feature, name)
            min, max = dict[feature]
            if min != max:
                data.iloc[:, feature] = (np.array(data.iloc[:,feature]) - min) / (max - min)
            print(np.unique(data.iloc[:, feature]))
    return data

columns = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root',
'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate']


# normalize train, validation and test sets
min_max = get_min_max('Data/{}/train'.format(train_origin))
train = pd.read_csv('Data/{}/train'.format(train_origin), header=None)
train = normalize(train, min_max, columns)
train.to_csv('Data/{}/train'.format(train_dest), header=False, index=False)

validation = pd.read_csv('Data/{}/validation'.format(test_origin), header=None)
validation = normalize(validation, min_max, columns)
validation.to_csv('Data/{}/validation'.format(test_dest), header=False, index=False)

test = pd.read_csv('Data/{}/corrected'.format(test_origin), header=None)
test = normalize(test, min_max, columns)
test.to_csv('Data/{}/corrected'.format(test_dest), header=False, index=False)
