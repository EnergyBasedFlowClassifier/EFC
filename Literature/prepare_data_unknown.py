import pandas as pd
from sklearn.model_selection import train_test_split

unknown = [
    "labeld_Botnet.csv",
    "labeld_Heartbleed-Port.csv",
    "labeld_DoS-Slowhttptest.csv",
    "labeld_Infiltration.csv",
    "labeld_WebAttack-XSS.csv",
    "labeld_WebAttack-BruteForce.csv",
    "labeld_WebAttack-SqlInjection.csv",
    "labeld_DoS-Slowloris.csv",
    "labeld_DoS-GlodenEye.csv",
    "labeld_SSH-Patator.csv",
]


web_attacks = [
    "labeld_WebAttack-XSS.csv",
    "labeld_WebAttack-BruteForce.csv",
    "labeld_WebAttack-SqlInjection.csv",
]

for path in unknown:
    data = pd.read_csv(f"Data/flow_labeled/{path}")
    data.drop(data.columns[0], inplace=True, axis=1)
    data.insert(len(data.columns), column="label", value=[0] * data.shape[0])

    if path in web_attacks:
        data.to_csv(f"Data/WebAttacks.csv", mode="a+", header=False, index=False)
    else:
        data.to_csv(f"Data/{path[7:]}", mode="a+", header=False, index=False)


sample_sizes = {
    "DoS slowloris": 10537,
    "Web Attack": 10537,
    "Heartbleed": 9859,
    "DoS Slowhttptest": 6786,
    "Infiltration": 5330,
    "Bot": 2075,
}

unknown = [
    "Bot",
    "Heartbleed",
    "DoS Slowhttptest",
    "Infiltration",
    "Web Attack",
    "DoS slowloris",
]

for path in unknown:
    data = pd.read_csv(f"Data_original/Pre_processed/{path}.csv")
    data.iloc[:, -1] = [0] * data.shape[0]
    data.iloc[:, -1] = pd.to_numeric(data.iloc[:, -1])

    if data.shape[0] > sample_sizes[path]:
        data = data.sample(sample_sizes[path], random_state=43)

    data.to_csv(f"Data_original/{path}.csv", mode="a+", header=False, index=False)
