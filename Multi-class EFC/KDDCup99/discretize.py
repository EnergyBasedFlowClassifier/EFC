import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
import random
import os

def get_normalization(file):
    intervals = []
    for feature in range(41):
        print(feature)
        data = pd.read_csv(file, usecols = [feature], header=None)
        data = list(data.iloc[:,0])
        intervals.append([min(data), max(data)])
        print(intervals[feature])
    return intervals

def normalize(data, dict):
    for feature in range(41):
        if feature not in [1,2,3]:
            min, max = dict[feature]
            if min != max:
                data.iloc[:, feature] = (np.array(data.iloc[:,feature]) - min) / (max - min)
                print(np.unique(data.iloc[:, feature]))
    return data

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
                quantiles = np.quantile(data, [i*(1/65) for i in range(1, 66)])
                quantiles = sorted(list(set([math.ceil(x) for x in quantiles])))
                intervals.append(quantiles)
            else:
                intervals.append(list(np.unique(data)))
        print(intervals[feature])
    return intervals


def discretize(data, dict):
    for feature in range(42):
        if feature in [1,2,3]:
            diff = np.setdiff1d(data.iloc[:, feature], dict[feature])
            if diff.shape[0] > 0:
                dict[feature] += [x for x in diff]
            for x, string in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value == string else value for value in data.iloc[:,feature]]
        else:
            if feature == 41:
                for idx in range(len(malicious_names)):
                    data.iloc[:, feature] = [idx if value in malicious_names[idx] else value for value in data.iloc[:,feature]]
                data.iloc[:, feature] = [100 if value not in range(len(malicious_names)) else value for value in data.iloc[:,feature]]
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

# remove duplicates from training
data = pd.read_csv('kddcup.data_10_percent', header=None)
print(data.shape[0])
data.drop_duplicates(inplace=True)
print(data.shape[0])
data.to_csv('kddcup.data_10_percent_unique', header=False, index=False)


#normalize train and test sets
# min_max = get_normalization('kddcup.data_10_percent_unique')
# train = pd.read_csv('kddcup.data_10_percent_unique', header=None)
# train = normalize(train, min_max)
# train.to_csv('kddcup.data_10_percent_normalized', header=False, index=False)
#
# test = pd.read_csv('corrected', header=None)
# test = normalize(test, min_max)
# test.to_csv('corrected_normalized', header=False, index=False)



#discretize train and test sets
intervals = get_intervals('kddcup.data_10_percent_unique')

reader = pd.read_csv('kddcup.data_10_percent_unique', chunksize=7000000, header=None)
for chunk in reader:
    data = discretize(chunk, intervals)
    data.drop(random.Random(2).sample(list(np.where(data.iloc[:, -1]==0)[0]), 30000), inplace=True)
    data.iloc[:, :-1].to_csv('X_kddcup.data_10_percent_discretized', mode='a', header=False, index=False)
    data.iloc[:, -1].to_csv('y_kddcup.data_10_percent_discretized', mode='a', header=False, index=False)

reader = pd.read_csv('corrected', chunksize=7000000, header=None)
for chunk in reader:
    data = discretize(chunk, intervals)
    data.iloc[:, :-1].to_csv('X_corrected_discretized', mode='a', header=False, index=False)
    data.iloc[:, -1].to_csv('y_corrected_discretized', mode='a', header=False, index=False)
