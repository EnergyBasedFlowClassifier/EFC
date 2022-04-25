import pandas as pd
from sklearn.model_selection import train_test_split

known = [
    "labeld_Monday-Benign.csv",
    "labeld_DoS-Hulk.csv",
    "labeld_DDoS.csv",
    "labeld_PortScan.csv",
    "labeld_FTP-Patator.csv",
]


for path in known:
    data = pd.read_csv(f"Data/flow_labeled/{path}")
    data.drop(data.columns[0], inplace=True, axis=1)
    data.insert(len(data.columns), column="label", value=[1] * data.shape[0])

    train, test = train_test_split(data, test_size=0.2, random_state=43)

    if path in ["labeld_DoS-Hulk.csv", "labeld_DDoS.csv", "labeld_PortScan.csv"]:
        train = train.sample(50000, random_state=43)
    print(path)

    print(train.shape)
    print(test.shape)

    train.to_csv("Data/train_known.csv", mode="a+", header=False, index=False)
    test.to_csv("Data/test_known.csv", mode="a+", header=False, index=False)
