import pandas as pd
import os
import shutil

os.makedirs("Data", exist_ok=True)
os.makedirs("Data/Normalized", exist_ok=True)
os.makedirs("Data/Normalized-Discretized", exist_ok=True)

shutil.move("kddcup.data_10_percent", "Data/kddcup.data_10_percent")
shutil.move("corrected", "Data/corrected")

# split train/validation
data = pd.read_csv('Data/kddcup.data_10_percent', header=None)
train = data.sample(frac=0.7, random_state=42)
validation = data.drop(train.index)
train.to_csv("Data/train", header=None, index=False)
validation.to_csv("Data/validation", header=None, index=False)
