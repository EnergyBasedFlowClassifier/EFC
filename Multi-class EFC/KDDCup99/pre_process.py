import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import math
from sklearn.model_selection import train_test_split
import os
import random

os.makedirs("Data", exist_ok=True)
os.makedirs("Data/Raw", exist_ok=True)
os.makedirs("Data/Encoded", exist_ok=True)
os.makedirs("Data/Encoded/Normalized", exist_ok=True)
os.makedirs("Data/Encoded/Discretized", exist_ok=True)
os.makedirs("Data/Encoded/Normalized-Discretized", exist_ok=True)
os.makedirs("Data/Encoded/Unique-Unknown", exist_ok=True)
os.makedirs("Data/Encoded/Unique-Unknown-Normalized", exist_ok=True)
os.makedirs("Data/Encoded/Unique-Unknown-Normalized-Discretized", exist_ok=True)

malicious_names = [['normal.'], ['back.', 'smurf.', 'teardrop.', 'neptune.', 'land.', 'pod.'],
['ipsweep.',  'portsweep.',  'satan.',  'nmap.'], ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.',
 'warezclient.', 'warezmaster.', 'phf.'], ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']]

# encode labels
train = pd.read_csv('Data/Raw/kddcup.data_10_percent', header=None)
test = pd.read_csv('Data/Raw/corrected', header=None)

for idx in range(len(malicious_names)):
    train.iloc[:, -1] = [idx if value in malicious_names[idx] else value for value in train.iloc[:,-1]]
    test.iloc[:, -1] = [idx if value in malicious_names[idx] else value for value in test.iloc[:,-1]]
test.iloc[:, -1] = [100 if value not in range(len(malicious_names)) else value for value in test.iloc[:,-1]]

train.to_csv('Data/Encoded/kddcup.data_10_percent', header=False, index=False)
test.to_csv('Data/Encoded/corrected', header=False, index=False)


# split train/validation
data = pd.read_csv('Data/Encoded/kddcup.data_10_percent', header=None)
train = data.sample(frac=0.7, random_state=42)
validation = data.drop(train.index)
train.to_csv("Data/Encoded/train", header=None, index=False)
validation.to_csv("Data/Encoded/validation", header=None, index=False)
