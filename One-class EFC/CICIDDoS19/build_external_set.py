import pandas as pd
import numpy as np
import random
import pandas as pd
import numpy as np
import random
import os
from zipfile import ZipFile

data_dis = np.load("Discretized_unique_CICIDS17.npy", allow_pickle=True)
unique, counts = np.unique(data_dis[:,-1], return_counts=True)
print(unique)
print(counts)
data_pre = np.array(pd.read_csv("Pre_processed.csv"))
for i in range(1,11):
    print(i)
    #separa conjuntos com cada tipo de amostra no conjunto discretizado e salva os indexes das amostras usadas
    normals = random.Random(i+15).sample([x for x in data_dis if x[-1] == 0], 10000)
    normals_index = [int(x[0]) for x in normals]
    DrDoS_DNS = random.Random(i+15).sample([x for x in data_dis if x[-1] == 1], 890)
    DrDoS_DNS_index = [int(x[0]) for x in DrDoS_DNS]
    DrDoS_LDAP = random.Random(i+15).sample([x for x in data_dis if x[-1] == 2], 370)
    DrDoS_LDAP_index = [int(x[0]) for x in DrDoS_LDAP]
    DrDoS_MSSQL = random.Random(i+15).sample([x for x in data_dis if x[-1] == 3], 890)
    DrDoS_MSSQL_index = [int(x[0]) for x in DrDoS_MSSQL]
    DrDoS_NetBIOS = random.Random(i+15).sample([x for x in data_dis if x[-1] == 4], 880)
    DrDoS_NetBIOS_index = [int(x[0]) for x in DrDoS_NetBIOS]
    DrDoS_NTP = random.Random(i+15).sample([x for x in data_dis if x[-1] == 5], 890)
    DrDoS_NTP_index = [int(x[0]) for x in DrDoS_NTP]
    DrDoS_SNMP = random.Random(i+15).sample([x for x in data_dis if x[-1] == 6], 200)
    DrDoS_SNMP_index = [int(x[0]) for x in DrDoS_SNMP]
    DrDoS_SSDP = random.Random(i+15).sample([x for x in data_dis if x[-1] == 7], 890)
    DrDoS_SSDP_index = [int(x[0]) for x in DrDoS_SSDP]
    DrDoS_UDP = random.Random(i+15).sample([x for x in data_dis if x[-1] == 8], 890)
    DrDoS_UDP_index = [int(x[0]) for x in DrDoS_UDP]
    Syn = random.Random(i+15).sample([x for x in data_dis if x[-1] == 9], 890)
    Syn_index = [int(x[0]) for x in Syn]
    TFTP = random.Random(i+15).sample([x for x in data_dis if x[-1] == 10], 890)
    TFTP_index = [int(x[0]) for x in TFTP]
    # UDP_lag = random.Random(6).sample([x for x in data_dis if x[-1] == 11], 390)
    # UDP_lag_index = [int(x[0]) for x in UDP_lag]
    # WebdDrDoS_DNS = random.Random(6).sample([x for x in data_dis if x[-1] == 12], 10)
    # WebdDrDoS_DNS_index = [int(x[0]) for x in WebdDrDoS_DNS]
    LDAP = random.Random(6).sample([x for x in data_dis if x[-1] == 13], 120)
    LDAP_index = [int(x[0]) for x in LDAP]
    NetBIOS = random.Random(6).sample([x for x in data_dis if x[-1] == 14], 150)
    NetBIOS_index = [int(x[0]) for x in NetBIOS]
    MSSQL = random.Random(6).sample([x for x in data_dis if x[-1] == 15], 630)
    MSSQL_index = [int(x[0]) for x in MSSQL]
    Portmap = random.Random(6).sample([x for x in data_dis if x[-1] == 16], 420)
    Portmap_index = [int(x[0]) for x in Portmap]
    UDP = random.Random(6).sample([x for x in data_dis if x[-1] == 17], 880)
    UDP_index = [int(x[0]) for x in UDP]
    UDP_lag = random.Random(6).sample([x for x in data_dis if x[-1] == 18], 120)
    UDP_lag_index = [int(x[0]) for x in UDP_lag]




    #separa conjuntos de cada tipo de amostra no conjunto não discretizado usando os indexes de cima
    DrDoS_DNS_pre = list(data_pre[[x for x in DrDoS_DNS_index], :])
    DrDoS_LDAP_pre = list(data_pre[[x for x in DrDoS_LDAP_index], :])
    DrDoS_MSSQL_pre = list(data_pre[[x for x in DrDoS_MSSQL_index], :])
    DrDoS_NetBIOS_pre = list(data_pre[[x for x in DrDoS_NetBIOS_index], :])
    DrDoS_NTP_pre = list(data_pre[[x for x in DrDoS_NTP_index], :])
    DrDoS_SNMP_pre = list(data_pre[[x for x in DrDoS_SNMP_index], :])
    DrDoS_SSDP_pre = list(data_pre[[x for x in DrDoS_SSDP_index], :])
    DrDoS_UDP_pre = list(data_pre[[x for x in DrDoS_UDP_index], :])
    Syn_pre = list(data_pre[[x for x in Syn_index], :])
    TFTP_pre = list(data_pre[[x for x in TFTP_index], :])
    LDAP_pre = list(data_pre[[x for x in LDAP_index], :])
    NetBIOS_pre = list(data_pre[[x for x in NetBIOS_index], :])
    MSSQL_pre = list(data_pre[[x for x in MSSQL_index], :])
    Portmap_pre = list(data_pre[[x for x in Portmap_index], :])
    UDP_pre = list(data_pre[[x for x in UDP_index], :])
    UDP_lag_pre = list(data_pre[[x for x in UDP_lag_index], :])
    normals_pre = list(data_pre[[x for x in normals_index], :])

    #Discretized test
    external_test = [x[1:-1] for x in normals[int(len(normals)*0.8)::] + DrDoS_DNS[int(len(DrDoS_DNS)*0.8)::] + DrDoS_LDAP[int(len(DrDoS_LDAP)*0.8)::]]
    external_test += [x[1:-1] for x in DrDoS_MSSQL[int(len(DrDoS_MSSQL)*0.8)::] + DrDoS_NetBIOS[int(len(DrDoS_NetBIOS)*0.8)::] + DrDoS_NTP[int(len(DrDoS_NTP)*0.8)::] + DrDoS_SNMP[int(len(DrDoS_SNMP)*0.8)::] + DrDoS_SSDP[int(len(DrDoS_SSDP)*0.8)::]]
    external_test += [x[1:-1] for x in DrDoS_UDP[int(len(DrDoS_UDP)*0.8)::] + Syn[int(len(Syn)*0.8)::] + TFTP[int(len(TFTP)*0.8)::] + LDAP[int(len(LDAP)*0.8)::] + NetBIOS[int(len(NetBIOS)*0.8)::] + MSSQL[int(len(MSSQL)*0.8)::]]
    external_test += [x[1:-1] for x in Portmap[int(len(Portmap)*0.8)::] + UDP[int(len(UDP)*0.8)::] + UDP_lag[int(len(UDP_lag)*0.8)::]]

    external_test_labels = [x[-1] for x in normals[int(len(normals)*0.8)::] + DrDoS_DNS[int(len(DrDoS_DNS)*0.8)::] + DrDoS_LDAP[int(len(DrDoS_LDAP)*0.8)::]]
    external_test_labels += [x[-1] for x in DrDoS_MSSQL[int(len(DrDoS_MSSQL)*0.8)::] + DrDoS_NetBIOS[int(len(DrDoS_NetBIOS)*0.8)::] + DrDoS_NTP[int(len(DrDoS_NTP)*0.8)::] + DrDoS_SNMP[int(len(DrDoS_SNMP)*0.8)::] + DrDoS_SSDP[int(len(DrDoS_SSDP)*0.8)::]]
    external_test_labels += [x[-1] for x in DrDoS_UDP[int(len(DrDoS_UDP)*0.8)::] + Syn[int(len(Syn)*0.8)::] + TFTP[int(len(TFTP)*0.8)::] + LDAP[int(len(LDAP)*0.8)::] + NetBIOS[int(len(NetBIOS)*0.8)::] + MSSQL[int(len(MSSQL)*0.8)::]]
    external_test_labels += [x[-1] for x in Portmap[int(len(Portmap)*0.8)::] + UDP[int(len(UDP)*0.8)::] + UDP_lag[int(len(UDP_lag)*0.8)::]]


    np.save("External_test/Discretized/Exp{}/external_test.npy".format(i), np.array(external_test))
    np.save("External_test/Discretized/Exp{}/external_test_labels.npy".format(i), np.array(external_test_labels))


    #Non_discretized test
    external_test = [x[1:-1] for x in normals_pre[int(len(normals)*0.8)::] + DrDoS_DNS_pre[int(len(DrDoS_DNS)*0.8)::] + DrDoS_LDAP_pre[int(len(DrDoS_LDAP)*0.8)::]]
    external_test += [x[1:-1] for x in DrDoS_MSSQL_pre[int(len(DrDoS_MSSQL)*0.8)::] + DrDoS_NetBIOS_pre[int(len(DrDoS_NetBIOS)*0.8)::] + DrDoS_NTP_pre[int(len(DrDoS_NTP)*0.8)::] + DrDoS_SNMP_pre[int(len(DrDoS_SNMP)*0.8)::] + DrDoS_SSDP_pre[int(len(DrDoS_SSDP)*0.8)::]]
    external_test += [x[1:-1] for x in DrDoS_UDP_pre[int(len(DrDoS_UDP)*0.8)::] + Syn_pre[int(len(Syn)*0.8)::] + TFTP_pre[int(len(TFTP)*0.8)::] + LDAP_pre[int(len(LDAP)*0.8)::] + NetBIOS_pre[int(len(NetBIOS)*0.8)::] + MSSQL_pre[int(len(MSSQL)*0.8)::]]
    external_test += [x[1:-1] for x in Portmap_pre[int(len(Portmap)*0.8)::] + UDP_pre[int(len(UDP)*0.8)::] + UDP_lag_pre[int(len(UDP_lag)*0.8)::]]

    external_test_labels = [x[-1] for x in normals_pre[int(len(normals)*0.8)::] + DrDoS_DNS_pre[int(len(DrDoS_DNS)*0.8)::] + DrDoS_LDAP_pre[int(len(DrDoS_LDAP)*0.8)::]]
    external_test_labels += [x[-1] for x in DrDoS_MSSQL_pre[int(len(DrDoS_MSSQL)*0.8)::] + DrDoS_NetBIOS_pre[int(len(DrDoS_NetBIOS)*0.8)::] + DrDoS_NTP_pre[int(len(DrDoS_NTP)*0.8)::] + DrDoS_SNMP_pre[int(len(DrDoS_SNMP)*0.8)::] + DrDoS_SSDP_pre[int(len(DrDoS_SSDP)*0.8)::]]
    external_test_labels += [x[-1] for x in DrDoS_UDP_pre[int(len(DrDoS_UDP)*0.8)::] + Syn_pre[int(len(Syn)*0.8)::] + TFTP_pre[int(len(TFTP)*0.8)::] + LDAP_pre[int(len(LDAP)*0.8)::] + NetBIOS_pre[int(len(NetBIOS)*0.8)::] + MSSQL_pre[int(len(MSSQL)*0.8)::]]
    external_test_labels += [x[-1] for x in Portmap_pre[int(len(Portmap)*0.8)::] + UDP_pre[int(len(UDP)*0.8)::] + UDP_lag_pre[int(len(UDP_lag)*0.8)::]]

    np.save("External_test/Non_discretized/Exp{}/external_test.npy".format(i), np.array(external_test))
    np.save("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(i), np.array(external_test_labels))
