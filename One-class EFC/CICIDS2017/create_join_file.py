import numpy as np
import pandas as pd

#this script joins all CICIDS17, except for 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX'
#and 'Wednesday-workingHours.pcap_ISCX', in one file (Joined_files) to be used in the experiments

def pre_processed():
    files = ['Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX',
    'Friday-WorkingHours-Morning.pcap_ISCX',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX',
     'Tuesday-WorkingHours.pcap_ISCX']


    concat = pd.read_csv('Pre_processed/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv')
    for file in files[1::]:
        data = pd.read_csv("Pre_processed/{}.csv".format(file))
        concat = pd.concat([concat, data], axis=0)

    concat.iloc[:, 0] = [x for x in range(0, concat.shape[0])]
    concat.to_csv("Pre_processed/Joined_files.csv", index=False)

pre_processed()
