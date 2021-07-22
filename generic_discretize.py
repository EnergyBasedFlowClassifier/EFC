import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  

#get discretization intervals based on the values
#​​of the training set. It also discretizes the training set.
def get_intervals(train_data, n_bins):
    intervals = []
    for feature in train_data.columns:
        col_values = np.array(train_data.loc[:,feature])
        if col_values.dtype in ['int', 'int64','int32']:
            train_data.loc[:,feature], retbins = pd.qcut(col_values, n_bins, labels=False, retbins=True, duplicates = 'drop')
            intervals.append(retbins.astype('int64'))
        elif col_values.dtype in ['float', 'float64', 'float32']:
            train_data.loc[:,feature], retbins = pd.qcut(col_values, n_bins, labels=False, retbins=True, duplicates = 'drop')
            intervals.append(retbins.astype('float64'))
        elif col_values.dtype == 'object':
            train_data.loc[:,feature] = col_values.astype('category').cat.codes
            intervals.append(np.array(col_values.astype('category').cat.categories))
        else: #if col_values.dtype not in ['int', 'int64'] and col_values.dtype not in ['float', 'float64'] and col_values.dtype != 'object':
            print("Column {} type not identified: {}".format(feature, col_values.dtype))
    return intervals, train_data


#discretizes data using predefined intervals (can be just one sample)
def discretize(test_data, intervals):
    for feature in range(test_data.shape[1]):
        col_values = test_data.loc[:,feature]
        print(feature)
        if col_values.dtype in ['int', 'int64']:
            test_data.loc[:,feature] = pd.cut(col_values, intervals[feature], labels=False, include_lowest=True, duplicates = 'drop')
        elif col_values.dtype in ['float', 'float64']:
            test_data.loc[:,feature] = pd.cut(col_values, intervals[feature], labels=False, include_lowest=True, duplicates = 'drop')
        elif col_values.dtype == 'object':
            test_data.loc[:,feature] = [np.where(intervals[feature] == x)[0] for x in col_values]
        else: #if col_values.dtype != 'int' and col_values.dtype != 'float' and col_values.dtype != 'object':
            print("Column {} type not identified {}:".format(feature, col_values.dtype))
            return test_data
    return test_data
