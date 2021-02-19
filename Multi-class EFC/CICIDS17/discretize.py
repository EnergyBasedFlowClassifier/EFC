import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype,  is_string_dtype
from matplotlib import pyplot as plt
import math


def get_intervals(columns):
    intervals = []
    for feature in range(len(columns)):
        print(feature)
        data = pd.read_csv("TrafficLabelling /Pre_processed_unique.csv", usecols = [feature], header=None)
        data = list(data.iloc[:,0])
        if feature in [2, len(columns)-1]:
            intervals.append(list(np.unique(data)))
        else:
            if len(np.unique(data)) > 10:
                quantiles = np.quantile(data, [0.03, 0.07, 0.1, 0.13, 0.17, 0.2, 0.23, 0.27, 0.3, 0.33, 0.37, 0.4, 0.43, 0.47, 0.5, 0.53, 0.57, 0.6, 0.63, 0.67, 0.7, 0.73, 0.77, 0.8, 0.83, 0.87, 0.9, 0.93, 0.97, 1.0])
                quantiles = sorted(list(set([math.ceil(x) for x in quantiles])))
                intervals.append(quantiles)
            else:
                intervals.append(list(np.unique(data)))
        print(intervals[feature])
    np.save("Dict_CICIDS17.npy", intervals)
    return intervals



def discretize(data, intervals, malicious_names):
    for feature in range(79):
        if feature == 2:
            for x, string in enumerate(intervals[feature]):
                data.iloc[:, feature] = [x if value == string else value for value in data.iloc[:,feature]]
        else:
            l_edge = np.min(data.iloc[:, feature])-1
            if max(data.iloc[:, feature]) > intervals[feature][-1]:
                intervals[feature].append(max(data.iloc[:, feature]))
            for x, r_edge in enumerate(intervals[feature]):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in data.iloc[:,feature]]
                l_edge = r_edge

    for i, value in enumerate(malicious_names):
        data.iloc[:,-1][data.iloc[:,-1] == value] = i

    print(np.unique(data.iloc[:,1:-1]))
    data.to_csv("TrafficLabelling /Discretized.csv", mode='a', header=False, index=False)


malicious_names = ['BENIGN',  'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack',
'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']



columns = ['SourcePort', 'DestinationPort', 'Protocol', 'FlowDuration',
'TotalFwdPackets', 'TotalBackwardPackets', 'TotalLengthofFwdPackets',
'TotalLengthofBwdPackets', 'FwdPacketLengthMax', 'FwdPacketLengthMin',
'FwdPacketLengthMean', 'FwdPacketLengthStd', 'BwdPacketLengthMax',
'BwdPacketLengthMin', 'BwdPacketLengthMean', 'BwdPacketLengthStd',
'FlowBytes-s', 'FlowPackets-s', 'FlowIATMean', 'FlowIATStd',
'FlowIATMax', 'FlowIATMin', 'FwdIATTotal', 'FwdIATMean', 'FwdIATStd',
'FwdIATMax', 'FwdIATMin', 'BwdIATTotal', 'BwdIATMean', 'BwdIATStd',
'BwdIATMax', 'BwdIATMin', 'FwdPSHFlags', 'BwdPSHFlags', 'FwdURGFlags',
'BwdURGFlags', 'FwdHeaderLength', 'BwdHeaderLength', 'FwdPackets-s',
'BwdPackets-s', 'MinPacketLength', 'MaxPacketLength',
'PacketLengthMean', 'PacketLengthStd', 'PacketLengthVariance',
'FINFlagCount', 'SYNFlagCount', 'RSTFlagCount', 'PSHFlagCount',
'ACKFlagCount', 'URGFlagCount', 'CWEFlagCount', 'ECEFlagCount',
'Down-UpRatio', 'AveragePacketSize', 'AvgFwdSegmentSize',
'AvgBwdSegmentSize', 'FwdAvgBytes-Bulk', 'FwdAvgPackets-Bulk',
'FwdAvgBulkRate', 'BwdAvgBytes-Bulk', 'BwdAvgPackets-Bulk',
'BwdAvgBulkRate', 'SubflowFwdPackets', 'SubflowFwdBytes',
'SubflowBwdPackets', 'SubflowBwdBytes', 'Init_Win_bytes_forward',
'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward',
'ActiveMean', 'ActiveStd', 'ActiveMax', 'ActiveMin', 'IdleMean',
'IdleStd', 'IdleMax', 'IdleMin', 'Label']


intervals = get_intervals(columns)
# intervals = np.load("Dict_CICIDS17.npy", allow_pickle=True)
reader = pd.read_csv("TrafficLabelling /Pre_processed_unique.csv", chunksize=2000000, header=None)
for chunk in reader:
    discretize(chunk, intervals, malicious_names)
