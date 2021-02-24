import numpy as np
import pandas as pd
import sys
import random
import os
from zipfile import ZipFile

def pre_process():
    files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX',
    'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX',
    'Friday-WorkingHours-Morning.pcap_ISCX',
    'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX',
    'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_fixed',
     'Tuesday-WorkingHours.pcap_ISCX',
     'Wednesday-workingHours.pcap_ISCX']


    concat = pd.read_csv('All_files_pre_processed/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv')

    for file in files[1::]:
        data = pd.read_csv("All_files_pre_processed/{}.csv".format(file))
        concat = pd.concat([concat, data], axis=0)

    concat.iloc[:, 0] = [x for x in range(0, concat.shape[0])]
    concat.to_csv("Pre_processed.csv", index=False)

pre_process()
