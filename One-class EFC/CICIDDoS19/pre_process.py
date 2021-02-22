import numpy as np
import pandas as pd
import sys
import shutil
import os
import math

def pre_process(day, files):
    for file in files:
        data = pd.read_csv("{}/{}_reduced.csv".format(day, file))

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

        data.drop(['Unnamed:0','FlowID','SourceIP','DestinationIP', 'Timestamp', 'FwdHeaderLength.1', 'SimillarHTTP','Inbound'], axis=1, inplace=True)
        data.dropna(axis=0, inplace=True)

        FB_max = 294400000000
        FP_max = 300000000
        data['FlowBytes-s'] = data['FlowBytes-s'].replace('Infinity', FB_max)
        data['FlowPackets-s'] = data['FlowPackets-s'].replace('Infinity', FP_max)
        data['FlowBytes-s'] = data['FlowBytes-s'].replace(np.inf, FB_max)
        data['FlowPackets-s'] = data['FlowPackets-s'].replace(np.inf, FP_max)

        for feature in data.columns:
            if data.loc[:, "{}".format(feature)].dtype == 'object' and feature not in ['Label', 'SimillarHTTP']:
                data.loc[:, "{}".format(feature)] = [str(x) for x in data.loc[:, "{}".format(feature)]]
                data.loc[:, "{}".format(feature)] = data.loc[:, "{}".format(feature)].str.replace(',','.')
                atribute_values = np.array(data.loc[:, "{}".format(feature)])
                data.loc[:, "{}".format(feature)] = atribute_values


        data.insert(0, 'Index', [x for x in range(data.shape[0])])
        data.to_csv("{}/{}_pre_processed.csv".format(day, file), index=False)


files_day1 = ['DrDoS_DNS','DrDoS_LDAP','DrDoS_MSSQL','DrDoS_NetBIOS','DrDoS_NTP','DrDoS_SNMP','DrDoS_SSDP','DrDoS_UDP','Syn','TFTP','UDPLag']
files_day2 = ['LDAP','MSSQL','NetBIOS','Portmap','Syn','UDP','UDPLag']

pre_process('01-12', files_day1)
pre_process('03-11', files_day2)
