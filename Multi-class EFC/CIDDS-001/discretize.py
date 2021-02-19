import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math

def get_intervals(columns):
    intervals = []
    for feature in range(len(columns)):
        print(feature)
        data = pd.read_csv("CIDDS-001/traffic/OpenStack/Pre_processed_unique.csv", usecols = [feature], header=None)
        data = list(data.iloc[:,0])
        if feature in [1,6, len(columns)-1]:
            intervals.append(list(np.unique(data)))
        else:
            if len(np.unique(data)) > 10:
                quantiles = np.quantile(data, [0.03, 0.07, 0.1, 0.13, 0.17, 0.2, 0.23, 0.27, 0.3, 0.33, 0.37, 0.4, 0.43, 0.47, 0.5, 0.53, 0.57, 0.6, 0.63, 0.67, 0.7, 0.73, 0.77, 0.8, 0.83, 0.87, 0.9, 0.93, 0.97, 1.0])
                quantiles = sorted(list(set([math.ceil(x) for x in quantiles])))
                intervals.append(quantiles)
            else:
                intervals.append(list(np.unique(data)))
        print(intervals[feature])
    np.save("Dict_CIDDS001.npy", intervals)
    return intervals


def discretize(data, dict, malicious_names):
    for feature in range(8):
        if feature in [1,6]:
            for x, string in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value == string else value for value in data.iloc[:,feature]]
        else:
            print(data.iloc[:, feature])
            l_edge = np.min(data.iloc[:, feature])-1
            if max(data.iloc[:, feature]) > dict[feature][-1]:
                dict[feature].append(max(data.iloc[:, feature]))
            for x, r_edge in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in data.iloc[:,feature]]
                l_edge = r_edge

    for i, value in enumerate(malicious_names):
        data.iloc[:,-1][data.iloc[:,-1] == value] = i

    print(np.unique(data.iloc[:,1:-1]))
    return data

columns = ['Duration','Proto','Src Pt','Dst Pt','Packets','Bytes','Flags','Tos','attackType']
malicious_names = ['normal','pingScan','bruteForce','portScan','dos']


intervals = get_intervals(columns)
# intervals = np.load("Dict_CIDDS001.npy", allow_pickle=True)
reader = pd.read_csv("CIDDS-001/traffic/OpenStack/Pre_processed_unique.csv", chunksize=6000000, header=None)
for chunk in reader:
    data = discretize(chunk, intervals, malicious_names)
    print(data[0].shape[0])
    print(np.unique(data[0]).shape[0])
    data.to_csv("CIDDS-001/traffic/OpenStack/Discretized.csv", mode='a', header=False, index=False)
