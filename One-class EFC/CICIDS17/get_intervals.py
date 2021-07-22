import numpy as np
import pandas as pd
import sys
from pandas.api.types import is_numeric_dtype,  is_string_dtype
from matplotlib import pyplot as plt
import math
from zipfile import ZipFile

def discretization_intervals():
    data = np.array(pd.read_csv("Pre_processed.csv"))
    dict = []
    for feature in range(1,data.shape[1]):
        print(feature)
        if len(np.unique(data[:,feature])) > 10 and feature != len(data[1])-1:
            list_normal = []
            list_malicious = []
            if feature in [16, 17]:
                copy = data[data[:,feature] != np.max(data[:,feature])]
                values = copy[:, feature]
                atribute_values = copy[:, [feature, len(data[1])-1]]
                quantiles = np.quantile(values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                quantiles = [math.ceil(x) for x in quantiles]
                quantiles = sorted(list(set(quantiles)))
                dict.append(quantiles)
            else:
                values = data[:, feature]
                atribute_values = data[:, [feature,len(data[1])-1]]
                quantiles = np.quantile(values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                quantiles = [math.ceil(x) for x in quantiles]
                quantiles = sorted(list(set(quantiles)))
                dict.append(quantiles)

        elif feature != len(data[1])-1:
            atribute_values = data[:, feature]
            dict.append(np.unique(atribute_values))

    return dict

#get discretization intervals for CICIDS17
dict = discretization_intervals()
np.save("Dict_CICIDS17.npy", dict)
