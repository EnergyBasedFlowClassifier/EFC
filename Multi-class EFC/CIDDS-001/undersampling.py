import pandas as pd
import numpy as np
import random
import os

chunksize = 5000000
malicious_names = ['normal','pingScan','bruteForce','portScan','dos']

for fold in range(1,6):
    print(fold)
    train_labels = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(fold), squeeze=True, header=None)
    print(np.unique(train_labels))

    normals_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[0])[0]), 6000)
    pingscan_index = list(np.where(train_labels==malicious_names[1])[0])
    bruteforce_index = list(np.where(train_labels==malicious_names[2])[0])
    portscan_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[3])[0]), 6000)
    dos_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[4])[0]), 6000)

    all_indexes = normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index
    all_indexes = np.array(all_indexes)

    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train.csv".format(fold), chunksize=chunksize, header=None)
    for i, chunk in enumerate(reader):

        indexes = list(all_indexes[np.isin(all_indexes, list(range(i*chunksize, (i+1)*chunksize)))])
        chunk_indexes = [x-(i*chunksize) for x in indexes]

        chunk.iloc[chunk_indexes, :].to_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(fold), mode='a', header=False, index=False)

        train_labels[indexes].to_csv("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(fold), mode='a', header=False, index=False)
