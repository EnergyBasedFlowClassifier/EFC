import numpy as np
import pandas as pd
import sys
from pandas.api.types import is_numeric_dtype,  is_string_dtype
from matplotlib import pyplot as plt
import math
from zipfile import ZipFile

intervals = []
for feature in range(1,80):
    print(feature)
    data = pd.read_csv("TrafficLabelling /Pre_processed.csv", usecols = [feature], header=None)
    data = list(data.iloc[:,0])
    if feature == 3:   #protocol - categorical
        intervals.append(list(np.unique(data)))
    else:
        if len(np.unique(data)) > 10:
            quantiles = np.quantile(data, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
            quantiles = sorted(list(set([math.ceil(x) for x in quantiles])))
            intervals.append(quantiles)
        else:
            intervals.append(list(np.unique(data)))
    print(intervals[feature-1])
np.save("Dict_CICIDS17.npy", intervals)
