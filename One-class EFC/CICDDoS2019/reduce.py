import numpy as np
import pandas as pd
import sys
import shutil
import os

#this script reduces the dataset files to a set with 10% of their original size

def reduce(files, day):
    for file in files:
        data = pd.read_csv("{}/{}.csv".format(day, file))
        labels = set(data.loc[:,' Label'])
        sample_size = int(data.shape[0]/10)
        new_file = pd.DataFrame([])
        for i in labels:
            if i == 'BENIGN':
                new_file = pd.concat([new_file, data[data[' Label'] == i]], axis=0)
            else:
                prop = list(data.loc[:,' Label']).count(i)/data.shape[0]
                size = int(prop*sample_size)
                atk_type = data[data[' Label'] == i]
                new_file = pd.concat([new_file, atk_type.sample(n=size, random_state=6)], axis=0)
        new_file = new_file.sample(frac=1)
        print(new_file.shape)
        new_file.to_csv("Reduced_files/{}_{}.csv".format(file, day), index=False)

files_day1 = ['DrDoS_DNS','DrDoS_LDAP','DrDoS_MSSQL','DrDoS_NetBIOS','DrDoS_NTP','DrDoS_SNMP','DrDoS_SSDP','DrDoS_UDP','Syn','TFTP','UDPLag']
files_day2 = ['LDAP','MSSQL','NetBIOS','Portmap','Syn','UDP','UDPLag']

reduce(files_day1, '01-12')
reduce(files_day2, '03-11')
