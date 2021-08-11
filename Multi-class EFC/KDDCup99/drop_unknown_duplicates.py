import numpy as np
import pandas as pd
import shutil

test = pd.read_csv('Data/Encoded/corrected', header=None)

unknown_idx = np.where(test.iloc[:, -1]==100)[0]
X_unknown = test.iloc[unknown_idx, :]

test.drop(unknown_idx, axis=0, inplace=True)

X_unknown.drop_duplicates(subset=X_unknown.columns[1:-1], inplace=True)

data = pd.concat([test, X_unknown], ignore_index=True)
data.to_csv("Data/Encoded/Unique-Unknown/corrected", header=False, index=False)

shutil.copyfile("Data/Encoded/validation","Data/Encoded/Unique-Unknown/validation")
