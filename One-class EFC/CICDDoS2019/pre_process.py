import numpy as np
import pandas as pd
import sys
import shutil
import os
import math

#this script pre proesses the reduced files from CICDDoS2019 dataset.
#it changes columns names, removing spaces and "/", and also removes some irrelevant columns such as identification columns.
#Samples with NaN values are removed and Infinity values are replaced by the column max value + 1.
#this script also add a index column in all files.

def pre_process(file):
    data = pd.read_csv("Reduced_files/{}.csv".format(file))

    data.columns = ['Unnamed:0','FlowID','SourceIP','SourcePort','DestinationIP','DestinationPort','Protocol','Timestamp',
    'FlowDuration','TotalFwdPackets','TotalBackwardPackets','TotalLengthofFwdPackets','TotalLengthofBwdPackets',
    'FwdPacketLengthMax','FwdPacketLengthMin','FwdPacketLengthMean','FwdPacketLengthStd','BwdPacketLengthMax',
    'BwdPacketLengthMin','BwdPacketLengthMean','BwdPacketLengthStd','FlowBytes-s','FlowPackets-s','FlowIATMean',
    'FlowIATStd','FlowIATMax','FlowIATMin','FwdIATTotal','FwdIATMean','FwdIATStd','FwdIATMax','FwdIATMin',
    'BwdIATTotal','BwdIATMean','BwdIATStd','BwdIATMax','BwdIATMin','FwdPSHFlags','BwdPSHFlags','FwdURGFlags',
    'BwdURGFlags','FwdHeaderLength','BwdHeaderLength','FwdPackets-s','BwdPackets-s','MinPacketLength','MaxPacketLength',
    'PacketLengthMean','PacketLengthStd','PacketLengthVariance','FINFlagCount','SYNFlagCount','RSTFlagCount','PSHFlagCount',
    'ACKFlagCount','URGFlagCount','CWEFlagCount','ECEFlagCount','Down-UpRatio','AveragePacketSize','AvgFwdSegmentSize',
    'AvgBwdSegmentSize','FwdHeaderLength.1','FwdAvgBytes-Bulk','FwdAvgPackets-Bulk','FwdAvgBulkRate','BwdAvgBytes-Bulk',
    'BwdAvgPackets-Bulk','BwdAvgBulkRate','SubflowFwdPackets','SubflowFwdBytes','SubflowBwdPackets','SubflowBwdBytes',
    'Init_Win_bytes_forward','Init_Win_bytes_backward','act_data_pkt_fwd','min_seg_size_forward','ActiveMean','ActiveStd','ActiveMax',
    'ActiveMin','IdleMean','IdleStd','IdleMax','IdleMin','SimillarHTTP','Inbound','Label']

    data.drop(['Unnamed:0','FlowID','SourceIP','DestinationIP', 'Timestamp', 'FwdHeaderLength.1','SimillarHTTP'], axis=1, inplace=True)
    data.dropna(axis=0, inplace=True)

    FB_max = 294400000000
    FP_max = 300000000
    data['FlowBytes-s'] = data['FlowBytes-s'].replace('Infinity', '{}'.format(FB_max))
    data['FlowPackets-s'] = data['FlowPackets-s'].replace('Infinity', '{}'.format(FP_max))
    data['FlowBytes-s'] = data['FlowBytes-s'].replace(np.inf, '{}'.format(FB_max))
    data['FlowPackets-s'] = data['FlowPackets-s'].replace(np.inf, '{}'.format(FP_max))

    for feature in data.columns:
        if data.loc[:, "{}".format(feature)].dtype == 'object' and feature != 'Label':
            data.loc[:, "{}".format(feature)] = [str(x) for x in data.loc[:, "{}".format(feature)]]
            data.loc[:, "{}".format(feature)] = data.loc[:, "{}".format(feature)].str.replace(',','.')
            atribute_values = np.array(data.loc[:, "{}".format(feature)])
            data.loc[:, "{}".format(feature)] = atribute_values

    data.insert(0, 'Index', [x for x in range(data.shape[0])])
    data.to_csv("Pre_processed/{}.csv".format(file), index=False)


files = ['DrDoS_DNS_01-12','DrDoS_LDAP_01-12','DrDoS_MSSQL_01-12','DrDoS_NetBIOS_01-12','DrDoS_NTP_01-12',
'DrDoS_SNMP_01-12','DrDoS_SSDP_01-12','DrDoS_UDP_01-12','Syn_01-12','TFTP_01-12','UDPLag_01-12',
'LDAP_03-11','MSSQL_03-11','NetBIOS_03-11','Portmap_03-11','Syn_03-11','UDP_03-11','UDPLag_03-11']

for file in files:
     pre_process(file)
