import numpy as np
import pandas as pd
import math
import sys
import os
from pandas.api.types import is_numeric_dtype
from zipfile import ZipFile

def discretize_data(dataset, intervals):
    data = pd.read_csv("Pre_processed.csv")
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

    malicious_names = ['BENIGN','DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
    'DrDoS_NTP', 'DrDoS_SNMP','DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'TFTP', 'UDP-lag',
    'WebDDoS', 'LDAP', 'NetBIOS', 'MSSQL', 'Portmap', 'UDP', 'UDPLag']
    for x, string in enumerate(malicious_names):
        print(x, string)
        data.iloc[:, -1] = [x if value == string else value for value in data.iloc[:,-1]]

    print(np.unique(data.iloc[:,1:-1]))
    data.to_csv("Discretized_{}.csv".format(dataset), header=False, index=False)


#discretize CICDDoS19 with both dataset discretization intervals
dict = np.load("Dict_CICDDoS19.npy", allow_pickle=True)
discretize_data("CICDDoS19", dict)
dict = np.load("../CICIDS17/Dict_CICIDS17.npy", allow_pickle=True)
discretize_data("CICIDS17", dict)
