import pandas as pd
import numpy as np

def pre_process_chunk(chunk):
    chunk['attackType'][chunk['attackType'] == '---'] = 'normal'

    chunk['Bytes'] = [str(x).replace(' ', '') for x in chunk['Bytes']]
    chunk['Bytes'] = [str(x).replace('.', '') for x in chunk['Bytes']]
    chunk['Bytes'] = [str(x).replace('M', '00000') for x in chunk['Bytes']]
    chunk['Bytes'] = [str(x).replace('K', '00') for x in chunk['Bytes']]
    chunk['Bytes'] = [float(x) for x in chunk['Bytes']]

    labels = chunk.columns
    for index in [0,3,5,9,12,14,15]:
        chunk.drop(labels[index], axis=1, inplace=True)

    return chunk


indexes = [0,0]
for week in [1,2,3,4]:
    reader = pd.read_csv("CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week{}.csv".format(week), chunksize=6000000)
    for chunk in reader:
        data = pre_process_chunk(chunk)
        print(chunk.shape[0])
        indexes[1] += chunk.shape[0]
        print(indexes)
        data.to_csv("CIDDS-001/traffic/OpenStack/Pre_processed.csv", mode='a', header=False, index=False)
        indexes[0] += chunk.shape[0]
