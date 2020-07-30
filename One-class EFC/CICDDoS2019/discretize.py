import numpy as np
import pandas as pd
import math
import sys
import os
from pandas.api.types import is_numeric_dtype

#this script performs the dataset discretization
#the first function creates discretization intervals using the entire database
#the second function uses the intervals to performe the discretizationfor each file. it also encodes labels in numbers

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


def discretize_data(file):
    dict = np.load("Dict.npy", allow_pickle=True)
    data = pd.read_csv("Pre_processed/{}.csv".format(file))
    for feature, intervals in enumerate(dict):
        if is_numeric_dtype(data.iloc[:,feature+1]):
            l_edge = np.min(data.iloc[:, feature + 1])-1
            for x, r_edge in enumerate(intervals):
                data.iloc[:, feature + 1] = [x if value > l_edge and value <= r_edge else value for value in data.iloc[:,feature + 1]]
                l_edge = r_edge
        else:
            for x, string in enumerate(intervals):
                data.iloc[:, feature + 1] = [x if value == string else value for value in data.iloc[:,feature+1]]

    data.loc[:, 'Label'] = [1 if value == "BENIGN" else 0 for value in data.loc[:, 'Label']]
    data.to_csv("Discretized/{}.csv".format(file), index=False)



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
'ActiveMin','IdleMean','IdleStd','IdleMax','IdleMin','Inbound']

files = ['DrDoS_DNS_01-12','DrDoS_LDAP_01-12','DrDoS_MSSQL_01-12','DrDoS_NetBIOS_01-12','DrDoS_NTP_01-12',
'DrDoS_SNMP_01-12','DrDoS_SSDP_01-12','DrDoS_UDP_01-12','Syn_01-12','TFTP_01-12','UDPLag_01-12',
'LDAP_03-11','MSSQL_03-11','NetBIOS_03-11','Portmap_03-11','Syn_03-11','UDP_03-11','UDPLag_03-11']


for feature in columns:
   dict = discretization_intervals(feature, files, dict)
np.save("Dict.npy", dict)

for file in files:
    discretize_data(file)
