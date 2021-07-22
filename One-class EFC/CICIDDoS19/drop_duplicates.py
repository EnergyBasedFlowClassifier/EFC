import numpy as np
import pandas as pd
import shutil
import os

def drop_duplicates(dict_dataset):
    data = pd.read_csv("Discretized_{}.csv".format(dict_dataset))
    data.drop_duplicates(subset=data.columns[1:-1], inplace=True)
    np.save("Discretized_unique_{}.npy".format(dict_dataset), data)

drop_duplicates("CICIDS17")
drop_duplicates("CICDDoS19")
