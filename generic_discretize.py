import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
pd.options.mode.chained_assignment = None

#get discretization intervals based on the values
#​​of the training set
def get_intervals(data, n_bins):
    intervals = []
    for feature in range(data.shape[1]):
        if is_numeric_dtype(data.iloc[:, feature]):
            _, retbins = pd.qcut(data.iloc[:, feature], n_bins, labels=False, retbins=True, duplicates = 'drop')
            intervals.append(retbins.astype('float64'))
        else:
            intervals.append(list(np.unique(data.iloc[:, feature])))

    return intervals

def discretize(data, intervals):
    for feature in range(data.shape[1]):
        col_values = data.iloc[:,feature]
        if is_numeric_dtype(col_values):
            data.iloc[:,feature] = pd.cut(col_values, intervals[feature], labels=False, include_lowest=True, duplicates = 'drop')
            data.iloc[:,feature].fillna(len(intervals[feature]), inplace=True)
        else:
            diff = np.setdiff1d(col_values, intervals[feature])
            if diff.shape[0] > 0:
                intervals[feature] += [x for x in diff]
            data.iloc[:,feature] = [intervals[feature].index(x) for x in col_values]
    return data.astype('int')
