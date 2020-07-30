import numpy as np
import pandas as pd

#this script removes the duplicate samples created after the discretization

def drop_duplicates(file):
    data = np.load("Discretized/{}.npy".format(file), allow_pickle=True)
    data = [list(x) for x in data]
    lista = [x[1::] for x in data]
    unique, idx = np.unique(lista, return_index=True, axis=0)
    data = [data[index] for index in idx]
    np.save("Discretized_unique/{}.npy".format(file), data)


files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX','Wednesday-workingHours.pcap_ISCX', 'Joined_files']
for file in files:
    drop_duplicates(file)
