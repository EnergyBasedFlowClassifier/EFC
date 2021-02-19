import pandas as pd
import numpy as np


def encode(chunk, proto, flags):
    for x, string in enumerate(proto):
        chunk.iloc[:, 1] = [x if value == string else value for value in chunk.iloc[:,1]]
    for x, string in enumerate(flags):
        chunk.iloc[:, 6] = [x if value == string else value for value in chunk.iloc[:,6]]
    return chunk

dict = np.load("Dict_CIDDS001.npy", allow_pickle=True)
proto, flags = dict[1], dict[6]
reader = pd.read_csv("CIDDS-001/traffic/OpenStack/Pre_processed_unique.csv", chunksize=6000000, header=None)
for chunk in reader:
    data = encode(chunk, proto, flags)
    data.to_csv("CIDDS-001/traffic/OpenStack/Pre_processed_encoded.csv", mode='a', header=False, index=False)
