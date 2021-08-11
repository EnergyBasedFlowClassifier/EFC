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
else:
    exit(-1)


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

# normalize train, validation and test sets
min_max = get_normalization('Data/{}/train'.format(train_origin))
train = pd.read_csv('Data/{}/train'.format(train_origin), header=None)
train = normalize(train, min_max)
train.to_csv('Data/{}/train'.format(train_dest), header=False, index=False)

validation = pd.read_csv('Data/{}/validation'.format(test_origin), header=None)
validation = normalize(validation, min_max)
validation.to_csv('Data/{}/validation'.format(test_dest), header=False, index=False)

test = pd.read_csv('Data/{}/corrected'.format(test_origin), header=None)
test = normalize(test, min_max)
test.to_csv('Data/{}/corrected'.format(test_dest), header=False, index=False)
