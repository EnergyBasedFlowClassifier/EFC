import numpy as np
import pandas as pd
import math
import sys
import os
from pandas.api.types import is_numeric_dtype
from zipfile import ZipFile

intervals = []
for feature in range(1,80):
    print(feature)
    data = pd.read_csv("Pre_processed.csv", usecols = [feature], header=None)
    data = list(data.iloc[:,0])
    if feature == 3:   #protocol - categorical
        intervals.append(list(np.unique(data)))
    else:
        if len(np.unique(data)) > 10:
            quantiles = np.quantile(data, [0.077, 0.153, 0.230, 0.307, 0.384, 0.461, 0.538, 0.616, 0.692, 0.769, 0.846, 0.923, 1])
            quantiles = sorted(list(set([math.ceil(x) for x in quantiles])))
            intervals.append(quantiles)
        else:
            intervals.append(list(np.unique(data)))
    print(intervals[feature-1])
np.save("Dict_CICDDoS19.npy", intervals)
