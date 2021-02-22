import numpy as np
import pandas as pd
import sys
from pandas.api.types import is_numeric_dtype,  is_string_dtype
from matplotlib import pyplot as plt
import math
from zipfile import ZipFile

def discretize(data, intervals, dataset_discretization):
    for feature in range(1,80):
        print(feature)
        if feature == 3: #protocol
            for x, string in enumerate(intervals[feature-1]):
                data.iloc[:, feature] = [x if value == string else value for value in data.iloc[:,feature]]
        else:
            l_edge = np.min(data.iloc[:, feature])-1
            if max(data.iloc[:, feature]) > intervals[feature-1][-1]:
                intervals[feature-1].append(max(data.iloc[:, feature]))
            for x, r_edge in enumerate(intervals[feature-1]):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in data.iloc[:,feature]]
                l_edge = r_edge

    malicious_names = ['BENIGN',  'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack Brute Force', 'Web Attack Sql Injection',  'Web Attack XSS',
     'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']
    for x, string in enumerate(malicious_names):
        print(x, string)
        data.iloc[:, -1] = [x if value == string else value for value in data.iloc[:,-1]]

    data.drop_duplicates(subset=data.columns[1:-1], inplace=True)
    print(np.unique(data.iloc[:,1:-1]))
    data.to_csv("GeneratedLabelledFlows/TrafficLabelling /Discretized_{}.csv".format(dataset_discretization), mode='a', header=False, index=False)



#discretize CICIDS17 with both dataset discretizations
intervals = np.load("Dict_CICIDS17.npy", allow_pickle=True)
reader = pd.read_csv("GeneratedLabelledFlows/TrafficLabelling /Pre_processed.csv", chunksize=500000, header=None)
for chunk in reader:
    discretize(chunk, intervals, "CICIDS17")

reader = pd.read_csv("GeneratedLabelledFlows/TrafficLabelling /Pre_processed.csv", chunksize=500000, header=None)
intervals = np.load("../CICIDDoS19/Dict_CICDDoS19.npy", allow_pickle=True)
for chunk in reader:
    discretize(chunk, intervals, "CICDDoS19")
