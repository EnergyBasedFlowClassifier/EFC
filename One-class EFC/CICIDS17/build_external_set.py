import pandas as pd
import numpy as np
import random
import os


data_disc = np.load("Discretized_unique_CICDDoS19.npy", allow_pickle=True)
unique, counts = np.unique(data_disc[:,-1], return_counts=True)
print(unique)
print(counts)
data_pre = np.array(pd.read_csv("Pre_processed.csv"))

for i in range(1,11):
    dos = random.Random(i+15).sample([x for x in data_disc if x[-1] == 1], 2910)
    dos_index = [int(x[0]) for x in dos]
    portscan = random.Random(i+15).sample([x for x in data_disc if x[-1] == 2], 1660)
    portscan_index = [int(x[0]) for x in portscan]
    bot = random.Random(i+15).sample([x for x in data_disc if x[-1] == 3], 110)
    bot_index = [int(x[0]) for x in bot]
    inf = random.Random(i+15).sample([x for x in data_disc if x[-1] == 4], 20)
    inf_index = [int(x[0]) for x in inf]
    bruteforce = random.Random(i+15).sample([x for x in data_disc if x[-1] == 5], 40)
    bruteforce_index = [int(x[0]) for x in bruteforce]
    sqlinj = random.Random(i+15).sample([x for x in data_disc if x[-1] == 6], 10)
    sqlinj_index = [int(x[0]) for x in sqlinj]
    xss = random.Random(i+15).sample([x for x in data_disc if x[-1] == 7], 10)
    xss_index = [int(x[0]) for x in xss]
    ftp = random.Random(i+15).sample([x for x in data_disc if x[-1] == 8], 150)
    ftp_index = [int(x[0]) for x in ftp]
    ssh = random.Random(i+15).sample([x for x in data_disc if x[-1] == 9], 60)
    ssh_index = [int(x[0]) for x in ssh]
    hulk = random.Random(i+15).sample([x for x in data_disc if x[-1] == 10], 2900)
    hulk_index = [int(x[0]) for x in hulk]
    goldeneye = random.Random(i+15).sample([x for x in data_disc if x[-1] == 11], 1800)
    goldeneye_index = [int(x[0]) for x in goldeneye]
    slowloris = random.Random(i+15).sample([x for x in data_disc if x[-1] == 12], 120)
    slowloris_index = [int(x[0]) for x in slowloris]
    slowhttp = random.Random(i+15).sample([x for x in data_disc if x[-1] == 13], 210)
    slowhttp_index = [int(x[0]) for x in slowhttp]
    normals = random.Random(i+15).sample([x for x in data_disc if x[-1] == 0], 10000)
    normals_index = [int(x[0]) for x in normals]


    dos_pre = list(data_pre[[x for x in dos_index], :])
    portscan_pre = list(data_pre[[x for x in portscan_index], :])
    bot_pre = list(data_pre[[x for x in bot_index], :])
    inf_pre = list(data_pre[[x for x in inf_index], :])
    bruteforce_pre = list(data_pre[[x for x in bruteforce_index], :])
    sqlinj_pre = list(data_pre[[x for x in sqlinj_index], :])
    xss_pre = list(data_pre[[x for x in xss_index], :])
    ftp_pre = list(data_pre[[x for x in ftp_index], :])
    ssh_pre = list(data_pre[[x for x in ssh_index], :])
    hulk_pre = list(data_pre[[x for x in hulk_index], :])
    goldeneye_pre = list(data_pre[[x for x in goldeneye_index], :])
    slowloris_pre = list(data_pre[[x for x in slowloris_index], :])
    slowhttp_pre = list(data_pre[[x for x in slowhttp_index], :])
    normals_pre = list(data_pre[[x for x in normals_index], :])


    external_test = [x[1:-1] for x in normals[int(len(normals)*0.8)::] + dos[int(len(dos)*0.8)::] + portscan[int(len(portscan)*0.8)::]]
    external_test += [x[1:-1] for x in bot[int(len(bot)*0.8)::] + inf[int(len(inf)*0.8)::] + bruteforce[int(len(bruteforce)*0.8)::] + sqlinj[int(len(sqlinj)*0.8)::] + xss[int(len(xss)*0.8)::]]
    external_test += [x[1:-1] for x in ftp[int(len(ftp)*0.8)::] + ssh[int(len(ssh)*0.8)::] + hulk[int(len(hulk)*0.8)::] + goldeneye[int(len(goldeneye)*0.8)::] + slowloris[int(len(slowloris)*0.8)::] + slowhttp[int(len(slowhttp)*0.8)::]]

    external_test_labels = [x[-1] for x in normals[int(len(normals)*0.8)::] + dos[int(len(dos)*0.8)::] + portscan[int(len(portscan)*0.8)::]]
    external_test_labels += [x[-1] for x in bot[int(len(bot)*0.8)::] + inf[int(len(inf)*0.8)::] + bruteforce[int(len(bruteforce)*0.8)::] + sqlinj[int(len(sqlinj)*0.8)::] + xss[int(len(xss)*0.8)::]]
    external_test_labels += [x[-1] for x in ftp[int(len(ftp)*0.8)::] + ssh[int(len(ssh)*0.8)::] + hulk[int(len(hulk)*0.8)::] + goldeneye[int(len(goldeneye)*0.8)::] + slowloris[int(len(slowloris)*0.8)::] + slowhttp[int(len(slowhttp)*0.8)::]]

    np.save("External_test/Discretized/Exp{}/external_test.npy".format(i), np.array(external_test,dtype=int))
    np.save("External_test/Discretized/Exp{}/external_test_labels.npy".format(i), np.array(external_test_labels,dtype=int))


    external_test = [x[1:-1] for x in normals_pre[int(len(normals)*0.8)::] + dos_pre[int(len(dos)*0.8)::] + portscan_pre[int(len(portscan)*0.8)::]]
    external_test += [x[1:-1] for x in bot_pre[int(len(bot)*0.8)::] + inf_pre[int(len(inf)*0.8)::] + bruteforce_pre[int(len(bruteforce)*0.8)::] + sqlinj_pre[int(len(sqlinj)*0.8)::] + xss_pre[int(len(xss)*0.8)::]]
    external_test += [x[1:-1] for x in ftp_pre[int(len(ftp)*0.8)::] + ssh_pre[int(len(ssh)*0.8)::] + hulk_pre[int(len(hulk)*0.8)::] + goldeneye_pre[int(len(goldeneye)*0.8)::] + slowloris_pre[int(len(slowloris)*0.8)::] + slowhttp_pre[int(len(slowhttp)*0.8)::]]

    external_test_labels = [x[-1] for x in normals_pre[int(len(normals)*0.8)::] + dos_pre[int(len(dos)*0.8)::] + portscan_pre[int(len(portscan)*0.8)::]]
    external_test_labels += [x[-1] for x in bot_pre[int(len(bot)*0.8)::] + inf_pre[int(len(inf)*0.8)::] + bruteforce_pre[int(len(bruteforce)*0.8)::] + sqlinj_pre[int(len(sqlinj)*0.8)::] + xss_pre[int(len(xss)*0.8)::]]
    external_test_labels += [x[-1] for x in ftp_pre[int(len(ftp)*0.8)::] + ssh_pre[int(len(ssh)*0.8)::] + hulk_pre[int(len(hulk)*0.8)::] + goldeneye_pre[int(len(goldeneye)*0.8)::] + slowloris_pre[int(len(slowloris)*0.8)::] + slowhttp_pre[int(len(slowhttp)*0.8)::]]

    np.save("External_test/Non_discretized/Exp{}/external_test.npy".format(i), np.array(external_test))
    np.save("External_test/Non_discretized/Exp{}/external_test_labels.npy".format(i), np.array(external_test_labels))
