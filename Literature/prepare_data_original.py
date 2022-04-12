import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


def pre_process(file):
    data = pd.read_csv(
        "../Multi-class EFC/CICIDS17/TrafficLabelling /{}.csv".format(file)
    )
    data.columns = [
        "FlowID",
        "SourceIP",
        "SourcePort",
        "DestinationIP",
        "DestinationPort",
        "Protocol",
        "Timestamp",
        "FlowDuration",
        "TotalFwdPackets",
        "TotalBackwardPackets",
        "TotalLengthofFwdPackets",
        "TotalLengthofBwdPackets",
        "FwdPacketLengthMax",
        "FwdPacketLengthMin",
        "FwdPacketLengthMean",
        "FwdPacketLengthStd",
        "BwdPacketLengthMax",
        "BwdPacketLengthMin",
        "BwdPacketLengthMean",
        "BwdPacketLengthStd",
        "FlowBytes-s",
        "FlowPackets-s",
        "FlowIATMean",
        "FlowIATStd",
        "FlowIATMax",
        "FlowIATMin",
        "FwdIATTotal",
        "FwdIATMean",
        "FwdIATStd",
        "FwdIATMax",
        "FwdIATMin",
        "BwdIATTotal",
        "BwdIATMean",
        "BwdIATStd",
        "BwdIATMax",
        "BwdIATMin",
        "FwdPSHFlags",
        "BwdPSHFlags",
        "FwdURGFlags",
        "BwdURGFlags",
        "FwdHeaderLength",
        "BwdHeaderLength",
        "FwdPackets-s",
        "BwdPackets-s",
        "MinPacketLength",
        "MaxPacketLength",
        "PacketLengthMean",
        "PacketLengthStd",
        "PacketLengthVariance",
        "FINFlagCount",
        "SYNFlagCount",
        "RSTFlagCount",
        "PSHFlagCount",
        "ACKFlagCount",
        "URGFlagCount",
        "CWEFlagCount",
        "ECEFlagCount",
        "Down-UpRatio",
        "AveragePacketSize",
        "AvgFwdSegmentSize",
        "AvgBwdSegmentSize",
        "FwdHeaderLength.1",
        "FwdAvgBytes-Bulk",
        "FwdAvgPackets-Bulk",
        "FwdAvgBulkRate",
        "BwdAvgBytes-Bulk",
        "BwdAvgPackets-Bulk",
        "BwdAvgBulkRate",
        "SubflowFwdPackets",
        "SubflowFwdBytes",
        "SubflowBwdPackets",
        "SubflowBwdBytes",
        "Init_Win_bytes_forward",
        "Init_Win_bytes_backward",
        "act_data_pkt_fwd",
        "min_seg_size_forward",
        "ActiveMean",
        "ActiveStd",
        "ActiveMax",
        "ActiveMin",
        "IdleMean",
        "IdleStd",
        "IdleMax",
        "IdleMin",
        "Label",
    ]

    if file != "Monday-WorkingHours.pcap_ISCX":
        data = data[data.Label != "BENIGN"]

    data.drop(
        ["FlowID", "SourceIP", "DestinationIP", "Timestamp", "FwdHeaderLength.1"],
        axis=1,
        inplace=True,
    )

    data.dropna(axis=0, inplace=True)

    data["FlowBytes-s"] = data["FlowBytes-s"].replace("Infinity", "2070000001")
    data["FlowPackets-s"] = data["FlowPackets-s"].replace("Infinity", "4000000")

    data["FlowBytes-s"][data["FlowBytes-s"] == np.inf] = "2070000001"
    data["FlowPackets-s"][data["FlowPackets-s"] == np.inf] = "4000000"

    for feature in data.columns:
        if data.loc[:, "{}".format(feature)].dtype == "object" and feature != "Label":
            data.loc[:, "{}".format(feature)] = [
                str(x) for x in data.loc[:, "{}".format(feature)]
            ]
            data.loc[:, "{}".format(feature)] = data.loc[
                :, "{}".format(feature)
            ].str.replace(",", ".")
            atribute_values = np.array(data.loc[:, "{}".format(feature)])
            data.loc[:, "{}".format(feature)] = atribute_values
            data["{}".format(feature)] = np.array(
                data["{}".format(feature)], dtype=np.float64
            )

    for i in np.unique(data.Label):
        samples = data[data.Label == i]
        samples.to_csv(
            f"Data_original/Pre_processed/{i}.csv",
            header=False,
            index=False,
        )





files = [
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX",
    "Friday-WorkingHours-Morning.pcap_ISCX",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX",
    "Tuesday-WorkingHours.pcap_ISCX",
    "Wednesday-workingHours.pcap_ISCX",
    "Monday-WorkingHours.pcap_ISCX",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_fixed",
]

with ProcessPoolExecutor() as executor:
    executor.map(pre_process, files)
