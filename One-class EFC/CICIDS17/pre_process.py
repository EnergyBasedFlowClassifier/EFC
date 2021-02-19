import numpy as np
import pandas as pd
import sys

def pre_process(file):
    data = pd.read_csv("TrafficLabelling/{}.csv".format(file))
    data.columns = ['FlowID', 'SourceIP', 'SourcePort', 'DestinationIP','DestinationPort', 'Protocol', 'Timestamp','FlowDuration',
    'TotalFwdPackets','TotalBackwardPackets',
    'TotalLengthofFwdPackets','TotalLengthofBwdPackets',
    'FwdPacketLengthMax','FwdPacketLengthMin',
    'FwdPacketLengthMean','FwdPacketLengthStd',
    'BwdPacketLengthMax','BwdPacketLengthMin',
    'BwdPacketLengthMean','BwdPacketLengthStd','FlowBytes-s',
    'FlowPackets-s','FlowIATMean','FlowIATStd','FlowIATMax',
    'FlowIATMin','FwdIATTotal','FwdIATMean','FwdIATStd',
    'FwdIATMax','FwdIATMin','BwdIATTotal','BwdIATMean',
    'BwdIATStd','BwdIATMax','BwdIATMin','FwdPSHFlags',
    'BwdPSHFlags','FwdURGFlags','BwdURGFlags',
    'FwdHeaderLength','BwdHeaderLength','FwdPackets-s',
    'BwdPackets-s','MinPacketLength','MaxPacketLength',
    'PacketLengthMean','PacketLengthStd','PacketLengthVariance',
    'FINFlagCount','SYNFlagCount','RSTFlagCount',
    'PSHFlagCount','ACKFlagCount','URGFlagCount',
    'CWEFlagCount','ECEFlagCount','Down-UpRatio',
    'AveragePacketSize','AvgFwdSegmentSize',
    'AvgBwdSegmentSize','FwdHeaderLength.1','FwdAvgBytes-Bulk',
    'FwdAvgPackets-Bulk','FwdAvgBulkRate','BwdAvgBytes-Bulk',
    'BwdAvgPackets-Bulk','BwdAvgBulkRate','SubflowFwdPackets',
    'SubflowFwdBytes','SubflowBwdPackets','SubflowBwdBytes',
    'Init_Win_bytes_forward','Init_Win_bytes_backward',
    'act_data_pkt_fwd','min_seg_size_forward','ActiveMean',
    'ActiveStd','ActiveMax','ActiveMin','IdleMean','IdleStd',
    'IdleMax','IdleMin','Label']

    data.drop(['FlowID','SourceIP','DestinationIP', 'Timestamp', 'FwdHeaderLength.1'], axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)


    data['FlowBytes-s'] = data['FlowBytes-s'].replace('Infinity', '2070000001')
    data['FlowPackets-s'] = data['FlowPackets-s'].replace('Infinity', '4000000')


    for feature in data.columns:
        if data.loc[:, "{}".format(feature)].dtype == 'object' and feature != 'Label':
            print("{}".format(feature))
            data.loc[:, "{}".format(feature)] = [str(x) for x in data.loc[:, "{}".format(feature)]]
            data.loc[:, "{}".format(feature)] = data.loc[:, "{}".format(feature)].str.replace(',','.')
            atribute_values = np.array(data.loc[:, "{}".format(feature)])
            data.loc[:, "{}".format(feature)] = atribute_values
            data["{}".format(feature)] = np.array(data["{}".format(feature)], dtype=np.float64)

    data.insert(0, 'Index', [x for x in range(0, data.shape[0])])
    print(data.head())
    data.to_csv("All_files_pre_processed/{}.csv".format(file), index=False)



files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX',
'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX',
'Friday-WorkingHours-Morning.pcap_ISCX',
'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX',
'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX',
'Tuesday-WorkingHours.pcap_ISCX',
'Wednesday-workingHours.pcap_ISCX',
'Monday-WorkingHours.pcap_ISCX']

for file in files:
    pre_process(file)
