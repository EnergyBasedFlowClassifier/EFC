import pandas as pd
import numpy as np
import random

chunksize = 2_000_000
malicious_names = ['BENIGN',  'DDoS', 'PortScan', 'Bot', 'Infiltration', 'Web Attack',
'FTP-Patator', 'SSH-Patator' , 'DoS Hulk', 'DoS GoldenEye',  'DoS slowloris', 'DoS Slowhttptest', 'Heartbleed']


for fold in range(1,6):
    train_labels = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(fold), squeeze=True, header=None)
    print(np.unique(train_labels))

    normals_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[0])[0]), 5000)
    dos_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[1])[0]), 5000)
    portscan_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[2])[0]), 5000)
    bot_index = list(np.where(train_labels==malicious_names[3])[0])
    inf_index = list(np.where(train_labels==malicious_names[4])[0])
    web_index = list(np.where(train_labels==malicious_names[5])[0])
    ftp_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[6])[0]), 5000)
    ssh_index = list(np.where(train_labels==malicious_names[7])[0])
    hulk_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[8])[0]), 5000)
    goldeneye_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[9])[0]), 5000)
    slowloris_index = list(np.where(train_labels==malicious_names[10])[0])
    slowhttp_index = list(np.where(train_labels==malicious_names[11])[0])
    heartbleed_index = list(np.where(train_labels==malicious_names[12])[0])

    all_indexes = normals_index + dos_index + portscan_index + bot_index + inf_index + web_index + ftp_index + ssh_index + hulk_index + goldeneye_index + slowloris_index + slowhttp_index + heartbleed_index
    all_indexes = np.array(all_indexes)

    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train.csv".format(fold), chunksize=chunksize, header=None)
    for i, chunk in enumerate(reader):

        indexes = list(all_indexes[np.isin(all_indexes, list(range(i*chunksize, (i+1)*chunksize)))])
        chunk_indexes = [x-(i*chunksize) for x in indexes]

        chunk.iloc[chunk_indexes, :].to_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), mode='a', header=False, index=False)

        train_labels[indexes].to_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(fold), mode='a', header=False, index=False)
