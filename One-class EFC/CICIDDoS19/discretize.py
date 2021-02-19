import numpy as np
import pandas as pd
import math
import sys
import os
from pandas.api.types import is_numeric_dtype
from zipfile import ZipFile

def discretization_intervals(feature, files, dict):
    feature_values = []
    for file in files:
        data = pd.read_csv("Pre_processed/{}.csv".format(file))
        feature_values += list(data['{}'.format(feature)])
    if is_numeric_dtype(feature_values[0]):
        if len(np.unique(feature_values)) > 10:
            quantiles = np.quantile(feature_values, [0.077, 0.153, 0.230, 0.307, 0.384, 0.461, 0.538, 0.616, 0.692, 0.769, 0.846, 0.923, 1])
            quantiles = [math.ceil(x) for x in quantiles]
            quantiles = sorted(list(set(quantiles)))
            if feature in ['FlowBytes-s', 'FlowPackets-s']:
                quantiles[-1] = quantiles[-1] - 1
                quantiles.append(max(feature_values))
            dict.append(quantiles)
        elif feature != 'Label':
            dict.append(np.unique(feature_values))
    else:
        dict.append(set(feature_values))
    return dict

def discretize_data(dict_dataset):
    malicious_names = ['BENIGN','DrDoS_DNS', 'DrDoS_LDAP', 'DrDoS_MSSQL', 'DrDoS_NetBIOS',
    'DrDoS_NTP', 'DrDoS_SNMP','DrDoS_SSDP', 'DrDoS_UDP', 'Syn', 'TFTP', 'UDP-lag',
    'WebDDoS', 'LDAP', 'NetBIOS', 'MSSQL', 'Portmap', 'UDP', 'UDPLag']
    data = pd.read_csv("Pre_processed.csv")
    dict = np.load("Dict_{}.npy".format(dict_dataset), allow_pickle=True)
    for feature in range(1,79):
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


columns = ['SourcePort','DestinationPort','Protocol','FlowDuration','TotalFwdPackets','TotalBackwardPackets','TotalLengthofFwdPackets','TotalLengthofBwdPackets',
'FwdPacketLengthMax','FwdPacketLengthMin','FwdPacketLengthMean','FwdPacketLengthStd','BwdPacketLengthMax',
'BwdPacketLengthMin','BwdPacketLengthMean','BwdPacketLengthStd','FlowBytes-s','FlowPackets-s','FlowIATMean',
'FlowIATStd','FlowIATMax','FlowIATMin','FwdIATTotal','FwdIATMean','FwdIATStd','FwdIATMax','FwdIATMin',
'BwdIATTotal','BwdIATMean','BwdIATStd','BwdIATMax','BwdIATMin','FwdPSHFlags','BwdPSHFlags','FwdURGFlags',
'BwdURGFlags','FwdHeaderLength','BwdHeaderLength','FwdPackets-s','BwdPackets-s','MinPacketLength','MaxPacketLength',
'PacketLengthMean','PacketLengthStd','PacketLengthVariance','FINFlagCount','SYNFlagCount','RSTFlagCount','PSHFlagCount',
'ACKFlagCount','URGFlagCount','CWEFlagCount','ECEFlagCount','Down-UpRatio','AveragePacketSize','AvgFwdSegmentSize',
'AvgBwdSegmentSize','FwdAvgBytes-Bulk','FwdAvgPackets-Bulk','FwdAvgBulkRate','BwdAvgBytes-Bulk',
'BwdAvgPackets-Bulk','BwdAvgBulkRate','SubflowFwdPackets','SubflowFwdBytes','SubflowBwdPackets','SubflowBwdBytes',
'Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','ActiveMean','ActiveStd','ActiveMax',
'ActiveMin','IdleMean','IdleStd','IdleMax','IdleMin']

#creates discretizaton intervals for CICDDoS19
dict = []
for feature in columns:
   dict = discretization_intervals(data, feature, dict)
np.save("Dict_CICDDoS19.npy", dict)

#drop duplicates before discretizaton
data = pd.read_csv("Pre_processed.csv")
data.drop_duplicates(subset=data.columns[1:-1], inplace=True)
data.iloc[:, 0] = [x for x in range(0, data.shape[0])]
data.to_csv("Pre_processed.csv", index=False)

#discretize CICDDoS19 with both dataset discretization intervals
discretize_data("CICIDS17")
discretize_data("CICDDoS19")
