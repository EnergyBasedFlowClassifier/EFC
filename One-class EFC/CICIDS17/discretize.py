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


def discretize_data(dict_dataset):
    malicious_names = ['BENIGN',  'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack Brute Force', 'Web Attack Sql Injection',  'Web Attack XSS',
     'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']
    dict = np.load("Dict_{}.npy".format(dict_dataset), allow_pickle=True)
    dict = [list(x) for x in dict]
    data = pd.read_csv("Pre_processed.csv")
    for feature in range(1,80):
        print(dict_dataset, feature)
        if is_numeric_dtype(data.iloc[:,feature]):
            l_edge = np.min(data.iloc[:, feature])-1
            if max(data.iloc[:, feature]) > dict[feature-1][-1]:
                dict[feature-1].append(max(data.iloc[:, feature]))
            for x, r_edge in enumerate(dict[feature-1]):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in data.iloc[:,feature]]
                l_edge = r_edge
        else:
            for x, string in enumerate(dict[feature-1]):
                data.iloc[:, feature] = [x if value == string else value for value in data.iloc[:,feature]]

    for i, value in enumerate(malicious_names):
        data.iloc[:,-1][data.iloc[:,-1] == value] = i

    data.to_csv("Discretized_{}.csv".format(dict_dataset), index=False)
    print(np.unique(data.iloc[:,1:-1]))

#get discretization intervals for CICIDS17
dict = discretization_intervals()
np.save("Dict_CICIDS17.npy", dict)

#drop duplicates before discretization
data = pd.read_csv("Pre_processed.csv")
data.drop_duplicates(subset=data.columns[1:-1], inplace=True)
data.iloc[:, 0] = [x for x in range(0, data.shape[0])]
data.to_csv("Pre_processed.csv", index=False)

#discretize
discretize_data("CICIDS17")
discretize_data("CICDDoS19")
