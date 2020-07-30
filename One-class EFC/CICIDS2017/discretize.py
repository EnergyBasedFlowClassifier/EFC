import numpy as np
import pandas as pd
import sys
from pandas.api.types import is_numeric_dtype,  is_string_dtype
from matplotlib import pyplot as plt
import math

#this script performs the dataset discretization
#the first function creates discretization intervals using the entire database
#the second function uses the intervals to performe the discretization. it also encodes label names in numbers

def discretization_intervals():
    data = np.array(pd.read_csv("Pre_processed.csv"))
    dict = {}
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
                dict[feature] = quantiles
            else:
                values = data[:, feature]
                atribute_values = data[:, [feature,len(data[1])-1]]
                quantiles = np.quantile(values, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
                quantiles = [math.ceil(x) for x in quantiles]
                quantiles = sorted(list(set(quantiles)))
                dict[feature] = quantiles

        elif feature != len(data[1])-1:
            atribute_values = data[:, feature]
            dict[feature] = np.unique(atribute_values)

    return dict


def discretize_data(file, features):
    malicious_names = ['BENIGN',  'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack Brute Force', 'Web Attack Sql Injection',  'Web Attack XSS',
     'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']
    data = np.array(pd.read_csv("Pre_processed/{}.csv".format(file)))
    for item in features.items():
        if np.max(data[:,item[0]]) > item[1][-1]:
            item[1][-1] = np.max(data[:,item[0]])
        l_edge = np.min(data[:, item[0]])-1
        for x, r_edge in enumerate(item[1]):
            data[:, item[0]] = [x if value > l_edge and value <= r_edge else value for value in data[:,item[0]]]
            l_edge = r_edge
        if item[0] in [16,17]:
            data[:, item[0]] = [len(item[1]) if value > item[1][-1] else value for value in data[:,item[0]]]

    for i, value in enumerate(malicious_names):
        data[:,-1][data[:,-1] == value] = i+1

    np.save("Discretized/{}.npy".format(file), data)


features = discretization_intervals()
files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX','Wednesday-workingHours.pcap_ISCX', "Joined_files"]
for file in files:
    discretize_data(file, features)
