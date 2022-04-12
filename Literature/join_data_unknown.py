import pandas as pd
import os

unknown_ocn = [
    "DoS-Slowloris",
    "Heartbleed-Port",
    "Infiltration",
    "Botnet",
    "DoS-Slowhttptest",
    "WebAttacks",
]

os.makedirs("Data/N_unknown_sets", exist_ok=True)

test_known = pd.read_csv("Data/test_known_binary.csv", header=None)

for i in range(1, 7):
    test_known.to_csv(
        f"Data/N_unknown_sets/N_{i}.csv", mode="a+", header=False, index=False
    )
    for path in unknown_ocn[:i]:
        test_unknown = pd.read_csv(f"Data/{path}.csv")

        test_unknown.to_csv(
            f"Data/N_unknown_sets/N_{i}.csv",
            mode="a+",
            header=False,
            index=False,
        )


unknown_original = [
    "DoS slowloris",
    "Heartbleed",
    "Infiltration",
    "Bot",
    "DoS Slowhttptest",
    "Web Attack",
]

os.makedirs("Data_original/N_unknown_sets", exist_ok=True)

test_known = pd.read_csv("Data_original/test_known_binary.csv", header=None)

for i in range(1, 7):
    test_known.to_csv(
        f"Data_original/N_unknown_sets/N_{i}.csv", mode="a+", header=False, index=False
    )

    for path in unknown_original[:i]:
        test_unknown = pd.read_csv(f"Data_original/{path}.csv")

        test_unknown.to_csv(
            f"Data_original/N_unknown_sets/N_{i}.csv",
            mode="a+",
            header=False,
            index=False,
        )
