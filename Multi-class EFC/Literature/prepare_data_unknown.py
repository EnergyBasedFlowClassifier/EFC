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
