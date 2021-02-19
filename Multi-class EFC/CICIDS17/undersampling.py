import pandas as pd
import numpy as np
import random

chunksize = 3000000

for sets in range(5):
    train_labels = np.array(pd.read_csv("5-fold_sets/Discretized/Sets{}/train_labels.csv".format(sets+1), header=None))
    print(np.unique(train_labels))

    normals_index = random.Random(1).sample(list(np.where(train_labels==0)[0]), 5000)
    dos_index = random.Random(1).sample(list(np.where(train_labels==1)[0]), 5000)
    portscan_index = random.Random(1).sample(list(np.where(train_labels==2)[0]), 5000)
    bot_index = list(np.where(train_labels==3)[0])
    inf_index = list(np.where(train_labels==4)[0])
    web_index = list(np.where(train_labels==5)[0])
    ftp_index = random.Random(1).sample(list(np.where(train_labels==6)[0]), 5000)
    ssh_index = list(np.where(train_labels==7)[0])
    hulk_index = random.Random(1).sample(list(np.where(train_labels==8)[0]), 5000)
    goldeneye_index = random.Random(1).sample(list(np.where(train_labels==9)[0]), 5000)
    slowloris_index = list(np.where(train_labels==10)[0])
    slowhttp_index = list(np.where(train_labels==11)[0])
    heartbleed_index = list(np.where(train_labels==12)[0])

    all_indexes = normals_index + dos_index + portscan_index + bot_index + inf_index + web_index + ftp_index + ssh_index + hulk_index + goldeneye_index + slowloris_index + slowhttp_index + heartbleed_index
    all_indexes = np.array(all_indexes)

    train_selected = []
    train_labels_selected = []
    reader = pd.read_csv("5-fold_sets/Discretized/Sets{}/train.csv".format(sets+1), chunksize=chunksize, header=None)
    for i, chunk in enumerate(reader):
        print(chunk.info())
        chunk = np.array(chunk)

        indexes = list(all_indexes[np.isin(all_indexes, list(range(i*chunksize, (i+1)*chunksize)))])
        chunk_indexes = [x-(i*chunksize) for x in indexes]

        train_selected += list(chunk[chunk_indexes])
        train_labels_selected += list(train_labels[indexes])

    np.savetxt("5-fold_sets/Discretized/Sets{}/reduced_train.csv".format(sets+1), train_selected, delimiter=',')
    np.savetxt("5-fold_sets/Discretized/Sets{}/reduced_train_labels.csv".format(sets+1), train_labels_selected, delimiter=',')


    #Non_discretized
    train_labels = np.array(pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(sets+1), header=None))
    print(np.unique(train_labels))
    train_selected = []
    train_labels_selected = []
    reader = pd.read_csv("5-fold_sets/Non_discretized/Sets{}/train.csv".format(sets+1), chunksize=chunksize, header=None)
    for i, chunk in enumerate(reader):
        chunk = np.array(chunk)

        indexes = list(all_indexes[np.isin(all_indexes, list(range(i*chunksize, (i+1)*chunksize)))])
        chunk_indexes = [x-(i*chunksize) for x in indexes]

        train_selected += list(chunk[chunk_indexes])
        train_labels_selected += list(train_labels[indexes])

    np.savetxt("5-fold_sets/Non_discretized/Sets{}/reduced_train.csv".format(sets+1), train_selected, delimiter=',')
    np.savetxt("5-fold_sets/Non_discretized/Sets{}/reduced_train_labels.csv".format(sets+1), train_labels_selected, delimiter=',')
