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

    train.to_csv("Data/train_known_binary.csv", mode="a+", header=False, index=False)
    test.to_csv("Data/test_known_binary.csv", mode="a+", header=False, index=False)


sample_sizes = {
    "BENIGN": 62639,
    "DDoS": 50000,
    "DoS Hulk": 50000,
    "PortScan": 50000,
    "FTP-Patator": 19941,
}

known = ["BENIGN", "DoS Hulk", "DDoS", "PortScan", "FTP-Patator"]

for path in known:
    data = pd.read_csv(f"Data_original/Pre_processed/{path}.csv")
    data.iloc[:, -1] = [1] * data.shape[0]
    data.iloc[:, -1] = pd.to_numeric(data.iloc[:, -1])

    train, test = train_test_split(data, test_size=0.2, random_state=43)

    if train.shape[0] > sample_sizes[path]:
        train = train.sample(50000, random_state=43)

    print(train.shape)
    print(test.shape)

    train.to_csv(
        "Data_original/train_known_binary.csv", mode="a+", header=False, index=False
    )
    test.to_csv(
        "Data_original/test_known_binary.csv", mode="a+", header=False, index=False
    )
