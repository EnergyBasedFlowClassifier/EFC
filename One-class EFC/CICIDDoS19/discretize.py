import numpy as np
import pandas as pd
import math
import sys
import os
from pandas.api.types import is_numeric_dtype
from zipfile import ZipFile


def discretize_data(dict_dataset):
    malicious_names = ['BENIGN','DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
    'DrDoS_NTP', 'DrDoS_SNMP','DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'TFTP', 'UDP-lag',
    'WebDDoS', 'LDAP', 'NetBIOS', 'MSSQL', 'Portmap', 'UDP', 'UDPLag']
    data = pd.read_csv("Pre_processed.csv")
    dict = list(np.load("Dict_{}.npy".format(dict_dataset), allow_pickle=True))
    for feature in range(1,80):
        print(dict_dataset, feature, dict[feature-1])
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



#drop duplicates before discretizaton
# data = pd.read_csv("Pre_processed.csv")
# data.drop_duplicates(subset=data.columns[1:-1], inplace=True)
# data.iloc[:, 0] = [x for x in range(0, data.shape[0])]
# data.to_csv("Pre_processed.csv", index=False)

#discretize CICDDoS19
discretize_data("CICDDoS19")
discretize_data("CICIDS17")
data = pd.read_csv("Discretized_CICIDS17.csv")
print(np.unique(data.iloc[:,1:-1]))
