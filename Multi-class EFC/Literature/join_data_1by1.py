import pandas as pd
import os
import numpy as np
import shutil

unknown = [
    "WebAttacks",
    "Botnet",
    "Heartbleed-Port",
    "DoS-Slowhttptest",
    "Infiltration",
    "DoS-Slowloris",
]

os.makedirs(f"Data/1by1_sets/", exist_ok=True)

test_known = pd.read_csv("Data/test_known.csv", header=None).values

for attack in unknown:
    test_unknown = pd.read_csv(f"Data/{attack}.csv", header=None).values

    test = np.concatenate((test_known, test_unknown))

    np.savetxt(f"Data/1by1_sets/{attack}.csv", test, delimiter=",")
