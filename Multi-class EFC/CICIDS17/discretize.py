import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype,  is_string_dtype
from matplotlib import pyplot as plt
import math


def get_intervals(file, columns):
    intervals = []
    for feature in range(len(columns)):
        print(feature)
        data = pd.read_csv(file, usecols = [feature], header=None)
        data = list(data.iloc[:,0])
        if feature in [2]:
            intervals.append(list(np.unique(data)))
        else:
            if len(np.unique(data)) > 10:
                quantiles = np.quantile(data, [0.03, 0.07, 0.1, 0.13, 0.17, 0.2, 0.23, 0.27, 0.3, 0.33, 0.37, 0.4, 0.43, 0.47, 0.5, 0.53, 0.57, 0.6, 0.63, 0.67, 0.7, 0.73, 0.77, 0.8, 0.83, 0.87, 0.9, 0.93, 0.97, 1.0])
                quantiles = sorted(list(set([math.ceil(x) for x in quantiles])))
                intervals.append(quantiles)
            else:
                intervals.append(list(np.unique(data)))
        print(intervals[feature])
    return intervals



def discretize(data, dict):
    for feature in range(79):
        if feature == 2:
            diff = np.setdiff1d(data.iloc[:, feature], dict[feature])
            if diff.shape[0] > 0:
                dict[feature] += [x for x in diff]
            for x, string in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value == string else value for value in data.iloc[:,feature]]
        else:
            l_edge = np.NINF
            for x, r_edge in enumerate(dict[feature]):
                data.iloc[:, feature] = [x if value > l_edge and value <= r_edge else value for value in data.iloc[:,feature]]
                if r_edge == dict[feature][-1]:
                    data.iloc[:, feature] = [x if value > r_edge else value for value in data.iloc[:,feature]]
                l_edge = r_edge

    return data


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
'IdleStd', 'IdleMax', 'IdleMin']

print(len(columns))

for fold in range(1,6):
    intervals = get_intervals("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), columns)
    np.save("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), intervals)

    # os.remove("5-fold_sets/Discretized/Sets{}/test.csv".format(fold))
    # os.remove("5-fold_sets/Discretized/Sets{}/train.csv".format(fold))

    # intervals = np.load("5-fold_sets/Discretized/Sets{}/Dict.npy".format(fold), allow_pickle=True)
    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), chunksize=4000000, header=None)
    for chunk in reader:
        data = discretize(chunk, intervals)
        data.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(fold), mode='a', header=False, index=False)

    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test.csv".format(fold), chunksize=4000000, header=None)
    for chunk in reader:
        data = discretize(chunk, intervals)
        data.to_csv("5-fold_sets/Discretized/Sets{}/test.csv".format(fold), mode='a', header=False, index=False)

    train_labels =  pd.read_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(fold), header=None)
    test_labels =  pd.read_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(fold), header=None)
    for i, value in enumerate(malicious_names):
        train_labels.iloc[:,-1][train_labels.iloc[:,-1] == value] = i
        test_labels.iloc[:,-1][test_labels.iloc[:,-1] == value] = i
    train_labels.to_csv("5-fold_sets/Discretized/Sets{}/reduced_train_labels.csv".format(fold), header=False, index=False)
    test_labels.to_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(fold), header=False, index=False)
