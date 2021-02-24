import numpy as np
import pandas as pd
import math
import sys
import os
from pandas.api.types import is_numeric_dtype
from zipfile import ZipFile

def discretization_intervals(feature, dict):
    data = pd.read_csv("Pre_processed.csv", usecols=[feature])
    feature_values = list(data.iloc[:, 0])
    print(feature)
    if feature == 'Protocol':
        print(np.unique(feature_values))
        dict.append(np.unique(feature_values))
    elif len(np.unique(feature_values)) > 10:
        quantiles = np.quantile(feature_values, [0.077, 0.153, 0.230, 0.307, 0.384, 0.461, 0.538, 0.616, 0.692, 0.769, 0.846, 0.923, 1])
        quantiles = [math.ceil(x) for x in quantiles]
        quantiles = sorted(list(set(quantiles)))
        if feature in ['FlowBytes-s', 'FlowPackets-s']:
            quantiles[-1] = quantiles[-1] - 1
            quantiles.append(max(feature_values))
        print(quantiles)
        dict.append(quantiles)
    else:
        print(np.unique(feature_values))
        dict.append(np.unique(feature_values))
    return dict

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
   dict = discretization_intervals(feature, dict)
np.save("Dict_CICDDoS19.npy", dict)
