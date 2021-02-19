import pandas as pd
import numpy as np


def encode(chunk, proto):
    for x, string in enumerate(proto):
        chunk.iloc[:, 2] = [x if value == string else value for value in chunk.iloc[:,2]]
    return chunk

dict = np.load("Dict_CICIDS17.npy", allow_pickle=True)
proto = dict[2]
reader = pd.read_csv("TrafficLabelling /Pre_processed_unique.csv", chunksize=4000000, header=None)
for chunk in reader:
    data = encode(chunk, proto)
    data.to_csv("TrafficLabelling /Pre_processed_encoded.csv", mode='a', header=False, index=False)
