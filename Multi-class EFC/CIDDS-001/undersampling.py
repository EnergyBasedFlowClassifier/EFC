import pandas as pd
import numpy as np
import random
import os

malicious_names = ['normal','pingScan','bruteForce','portScan','dos']

for fold in range(1,6):
    print(fold)
    train = pd.read_csv("5-fold_sets/Raw/Sets{}/train.csv".format(fold), header=None)
    train_labels = train.iloc[:, -1]

    normals_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[0])[0]), 6000)
    pingscan_index = list(np.where(train_labels==malicious_names[1])[0])
    bruteforce_index = list(np.where(train_labels==malicious_names[2])[0])
    portscan_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[3])[0]), 6000)
    dos_index = random.Random(1).sample(list(np.where(train_labels==malicious_names[4])[0]), 6000)

    all_indexes = normals_index + pingscan_index + bruteforce_index + portscan_index + dos_index
    all_indexes = np.array(all_indexes)

    train.iloc[all_indexes, :].to_csv("5-fold_sets/Raw/Sets{}/reduced_train.csv".format(fold), mode='a', header=False, index=False)
