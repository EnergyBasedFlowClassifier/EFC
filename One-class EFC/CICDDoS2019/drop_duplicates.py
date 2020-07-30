import numpy as np
import pandas as pd
import shutil
import os

#this script removes the duplicate samples created after the discretization
#for the files used in the experiments

def drop_duplicates(file):
    data = np.array(pd.read_csv("Discretized_{}.csv".format(file)))
    data = [list(x) for x in data]
    lista = [x[1:-1] for x in data]
    unique, idx = np.unique(lista, return_index=True, axis=0)
    data = [data[index] for index in idx]
    np.save("Discretized_unique/{}.npy".format(file), data)

files = ['DrDoS_NTP_01-12','TFTP_01-12','Syn_03-11' ]
for file in files:
    drop_duplicates(file)
