import numpy as np
import pandas as pd
import sys
import shutil
import os
# import psutil

#this script reduces the dataset files to a set with 20% of their original size

def reduce(files, day):
    for file in files:
        print(day,file)
        counter = 0
        counter_benign = 0
        with open("{}/{}.csv".format(day, file), "r") as fl:
            for line in fl:
                raw = line.split(",")
                a = tuple(np.array(raw))
                with open("Reduced/{}/{}.csv".format(day, file), "a") as fl2:
                    if len(raw) > 0 and raw[-1]=='BENIGN\n':
                        counter_benign += 1
                        for word in a:
                            fl2.write(word + ",")
                    if len(raw) > 0 and raw[-1]=="{}\n".format(file) and counter < 70000:
                        counter += 1
                        for word in a:
                            fl2.write(word + ",")
                    if counter_benign > 60000 and counter >= 70000:
                        break

files_day1 = ['DrDoS_DNS','DrDoS_LDAP','DrDoS_MSSQL','DrDoS_NetBIOS','DrDoS_NTP','DrDoS_SNMP','DrDoS_SSDP','DrDoS_UDP','Syn','TFTP','UDPLag']
files_day2 = ['LDAP','MSSQL','NetBIOS','Portmap','Syn','UDP','UDPLag']

reduce(files_day1, '01-12')
reduce(files_day2, '03-11')
