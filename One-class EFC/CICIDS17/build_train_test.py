import pandas as pd
import numpy as np
import random
import os

data_disc = np.load("Discretized_unique_CICIDS17.npy", allow_pickle=True)
data_pre = np.array(pd.read_csv("Pre_processed.csv"))

for i in range(1,11):
    dos = random.Random(i).sample([x for x in data_disc if x[-1] == 1], 2740)
    dos_index = [int(x[0]) for x in dos]
    portscan = random.Random(i).sample([x for x in data_disc if x[-1] == 2], 1060)
    portscan_index = [int(x[0]) for x in portscan]
    bot = random.Random(i).sample([x for x in data_disc if x[-1] == 3], 110)
    bot_index = [int(x[0]) for x in bot]
    inf = random.Random(i).sample([x for x in data_disc if x[-1] == 4], 20)
    inf_index = [int(x[0]) for x in inf]
    bruteforce = random.Random(i).sample([x for x in data_disc if x[-1] == 5], 50)
    bruteforce_index = [int(x[0]) for x in bruteforce]
    sqlinj = random.Random(i).sample([x for x in data_disc if x[-1] == 6], 10)
    sqlinj_index = [int(x[0]) for x in sqlinj]
    xss = random.Random(i).sample([x for x in data_disc if x[-1] == 7], 10)
    xss_index = [int(x[0]) for x in xss]
    ftp = random.Random(i).sample([x for x in data_disc if x[-1] == 8], 170)
    ftp_index = [int(x[0]) for x in ftp]
    ssh = random.Random(i).sample([x for x in data_disc if x[-1] == 9], 80)
    ssh_index = [int(x[0]) for x in ssh]
    hulk = random.Random(i).sample([x for x in data_disc if x[-1] == 10], 2730)
    hulk_index = [int(x[0]) for x in hulk]
    goldeneye = random.Random(i).sample([x for x in data_disc if x[-1] == 11], 2730)
    goldeneye_index = [int(x[0]) for x in goldeneye]
    slowloris = random.Random(i).sample([x for x in data_disc if x[-1] == 12], 120)
    slowloris_index = [int(x[0]) for x in slowloris]
    slowhttp = random.Random(i).sample([x for x in data_disc if x[-1] == 13], 170)
    slowhttp_index = [int(x[0]) for x in slowhttp]
    normals = random.Random(i).sample([x for x in data_disc if x[-1] == 0], 10000)
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


    #Discretized sets
    test = [x[1:-1] for x in normals[int(len(normals)*0.8)::] + dos[int(len(dos)*0.8)::] + portscan[int(len(portscan)*0.8)::]]
    test += [x[1:-1] for x in bot[int(len(bot)*0.8)::] + inf[int(len(inf)*0.8)::] + bruteforce[int(len(bruteforce)*0.8)::] + sqlinj[int(len(sqlinj)*0.8)::] + xss[int(len(xss)*0.8)::]]
    test += [x[1:-1] for x in ftp[int(len(ftp)*0.8)::] + ssh[int(len(ssh)*0.8)::] + hulk[int(len(hulk)*0.8)::] + goldeneye[int(len(goldeneye)*0.8)::] + slowloris[int(len(slowloris)*0.8)::] + slowhttp[int(len(slowhttp)*0.8)::]]

    test_labels = [x[-1] for x in normals[int(len(normals)*0.8)::] + dos[int(len(dos)*0.8)::] + portscan[int(len(portscan)*0.8)::]]
    test_labels += [x[-1] for x in bot[int(len(bot)*0.8)::] + inf[int(len(inf)*0.8)::] + bruteforce[int(len(bruteforce)*0.8)::] + sqlinj[int(len(sqlinj)*0.8)::] + xss[int(len(xss)*0.8)::]]
    test_labels += [x[-1] for x in ftp[int(len(ftp)*0.8)::] + ssh[int(len(ssh)*0.8)::] + hulk[int(len(hulk)*0.8)::] + goldeneye[int(len(goldeneye)*0.8)::] + slowloris[int(len(slowloris)*0.8)::] + slowhttp[int(len(slowhttp)*0.8)::]]

    train = [x[1:-1] for x in normals[:int(len(normals)*0.8)] + dos[:int(len(dos)*0.8)] + portscan[:int(len(portscan)*0.8)]]
    train += [x[1:-1] for x in bot[:int(len(bot)*0.8)] + inf[:int(len(inf)*0.8)] + bruteforce[:int(len(bruteforce)*0.8)] + sqlinj[:int(len(sqlinj)*0.8)] + xss[:int(len(xss)*0.8)]]
    train += [x[1:-1] for x in ftp[:int(len(ftp)*0.8)] + ssh[:int(len(ssh)*0.8)] + hulk[:int(len(hulk)*0.8)] + goldeneye[:int(len(goldeneye)*0.8)] + slowloris[:int(len(slowloris)*0.8)] + slowhttp[:int(len(slowhttp)*0.8)]]

    train_labels = [x[-1] for x in normals[:int(len(normals)*0.8)] + dos[:int(len(dos)*0.8)] + portscan[:int(len(portscan)*0.8)]]
    train_labels += [x[-1] for x in bot[:int(len(bot)*0.8)] + inf[:int(len(inf)*0.8)] + bruteforce[:int(len(bruteforce)*0.8)] + sqlinj[:int(len(sqlinj)*0.8)] + xss[:int(len(xss)*0.8)]]
    train_labels += [x[-1] for x in ftp[:int(len(ftp)*0.8)] + ssh[:int(len(ssh)*0.8)] + hulk[:int(len(hulk)*0.8)] + goldeneye[:int(len(goldeneye)*0.8)] + slowloris[:int(len(slowloris)*0.8)] + slowhttp[:int(len(slowhttp)*0.8)]]

    np.save("Data/Discretized/Exp{}/train.npy".format(i), np.array(train))
    np.save("Data/Discretized/Exp{}/train_labels.npy".format(i), np.array(train_labels))
    np.save("Data/Discretized/Exp{}/test.npy".format(i), np.array(test))
    np.save("Data/Discretized/Exp{}/test_labels.npy".format(i), np.array(test_labels))


    #Non_discretized sets
    test = [x[1:-1] for x in normals_pre[int(len(normals)*0.8)::] + dos_pre[int(len(dos)*0.8)::] + portscan_pre[int(len(portscan)*0.8)::]]
    test += [x[1:-1] for x in bot_pre[int(len(bot)*0.8)::] + inf_pre[int(len(inf)*0.8)::] + bruteforce_pre[int(len(bruteforce)*0.8)::] + sqlinj_pre[int(len(sqlinj)*0.8)::] + xss_pre[int(len(xss)*0.8)::]]
    test += [x[1:-1] for x in ftp_pre[int(len(ftp)*0.8)::] + ssh_pre[int(len(ssh)*0.8)::] + hulk_pre[int(len(hulk)*0.8)::] + goldeneye_pre[int(len(goldeneye)*0.8)::] + slowloris_pre[int(len(slowloris)*0.8)::] + slowhttp_pre[int(len(slowhttp)*0.8)::]]

    test_labels = [x[-1] for x in normals_pre[int(len(normals)*0.8)::] + dos_pre[int(len(dos)*0.8)::] + portscan_pre[int(len(portscan)*0.8)::]]
    test_labels += [x[-1] for x in bot_pre[int(len(bot)*0.8)::] + inf_pre[int(len(inf)*0.8)::] + bruteforce_pre[int(len(bruteforce)*0.8)::] + sqlinj_pre[int(len(sqlinj)*0.8)::] + xss_pre[int(len(xss)*0.8)::]]
    test_labels += [x[-1] for x in ftp_pre[int(len(ftp)*0.8)::] + ssh_pre[int(len(ssh)*0.8)::] + hulk_pre[int(len(hulk)*0.8)::] + goldeneye_pre[int(len(goldeneye)*0.8)::] + slowloris_pre[int(len(slowloris)*0.8)::] + slowhttp_pre[int(len(slowhttp)*0.8)::]]

    train = [x[1:-1] for x in normals_pre[:int(len(normals)*0.8)] + dos_pre[:int(len(dos)*0.8)] + portscan_pre[:int(len(portscan)*0.8)]]
    train += [x[1:-1] for x in bot_pre[:int(len(bot)*0.8)] + inf_pre[:int(len(inf)*0.8)] + bruteforce_pre[:int(len(bruteforce)*0.8)] + sqlinj_pre[:int(len(sqlinj)*0.8)] + xss_pre[:int(len(xss)*0.8)]]
    train += [x[1:-1] for x in ftp_pre[:int(len(ftp)*0.8)] + ssh_pre[:int(len(ssh)*0.8)] + hulk_pre[:int(len(hulk)*0.8)] + goldeneye_pre[:int(len(goldeneye)*0.8)] + slowloris_pre[:int(len(slowloris)*0.8)] + slowhttp_pre[:int(len(slowhttp)*0.8)]]

    train_labels = [x[-1] for x in normals_pre[:int(len(normals)*0.8)] + dos_pre[:int(len(dos)*0.8)] + portscan_pre[:int(len(portscan)*0.8)]]
    train_labels += [x[-1] for x in bot_pre[:int(len(bot)*0.8)] + inf_pre[:int(len(inf)*0.8)] + bruteforce_pre[:int(len(bruteforce)*0.8)] + sqlinj_pre[:int(len(sqlinj)*0.8)] + xss_pre[:int(len(xss)*0.8)]]
    train_labels += [x[-1] for x in ftp_pre[:int(len(ftp)*0.8)] + ssh_pre[:int(len(ssh)*0.8)] + hulk_pre[:int(len(hulk)*0.8)] + goldeneye_pre[:int(len(goldeneye)*0.8)] + slowloris_pre[:int(len(slowloris)*0.8)] + slowhttp_pre[:int(len(slowhttp)*0.8)]]

    np.save("Data/Non_discretized/Exp{}/train.npy".format(i), np.array(train))
    np.save("Data/Non_discretized/Exp{}/train_labels.npy".format(i), np.array(train_labels))
    np.save("Data/Non_discretized/Exp{}/test.npy".format(i), np.array(test))
    np.save("Data/Non_discretized/Exp{}/test_labels.npy".format(i), np.array(test_labels))
