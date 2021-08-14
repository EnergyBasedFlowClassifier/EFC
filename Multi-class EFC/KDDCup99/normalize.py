import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
from sklearn.model_selection import train_test_split
import os
import random
import sys

def get_normalization(file):
    intervals = []
    for feature in range(41):
        data = pd.read_csv(file, usecols = [feature], header=None, squeeze=True)
        intervals.append([min(data), max(data)])
    return intervals

def normalize(data, dict):
    for feature in range(41):
        if is_numeric_dtype(data.iloc[:, feature]):
            min, max = dict[feature]
            if min != max:
                data.iloc[:, feature] = (np.array(data.iloc[:,feature]) - min) / (max - min)
    return data

# get min and max from train
min_max = get_normalization('Data/train')

# load data
train = pd.read_csv('Data/train', header=None)
validation = pd.read_csv('Data/validation', header=None)
test = pd.read_csv('Data/corrected', header=None)

# normalize train, validation and test sets
train = normalize(train, min_max)
validation = normalize(validation, min_max)
test = normalize(test, min_max)

# save
train.to_csv('Data/Normalized/train', header=False, index=False)
validation.to_csv('Data/Normalized/validation', header=False, index=False)
test.to_csv('Data/Normalized/corrected', header=False, index=False)
