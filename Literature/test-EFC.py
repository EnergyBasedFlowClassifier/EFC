import pandas as pd
import random
from sklearn.model_selection import train_test_split
import os

known = ["labeld_Monday-Benign.csv",
"labeld_DoS-Hulk.csv",
"labeld_DDoS.csv",
"labeld_PortScan.csv",
"labeld_FTP-Patator.csv"]

if os.path.exists("Data/train_known"):
    os.remove("Data/train_known")
if os.path.exists("Data/test_known"):
    os.remove("Data/test_known")
if os.path.exists("Data/test_unknown"):
    os.remove("Data/test_unknown")

for path, i in zip(known, range(5)):
    data = pd.read_csv(f"Data/flow_labeled/{path}")
    data.drop(data.columns[0], inplace=True, axis=1)
    data.insert(len(data.columns), column="label", value=[i]*data.shape[0])
    train, test = train_test_split(data, test_size=0.2, random_state=43)
 
    if path in ["labeld_DoS-Hulk.csv","labeld_DDoS.csv","labeld_PortScan.csv"]:
        train = train.sample(50000, random_state=43)
    
    print(path)
    print(train.shape)
    print(test.shape)

    train.to_csv("Data/train_known", mode="a+", header=False, index=False)
    test.to_csv("Data/test_known", mode="a+", header=False, index=False)


#"labeld_SSH-Patator.csv",
#"labeld_DoS-GlodenEye.csv",

unknown = ["labeld_WebAttack-SqlInjection.csv",
"labeld_Botnet.csv",
"labeld_Heartbleed-Port.csv",
"labeld_DoS-Slowhttptest.csv",
"labeld_Infiltration.csv",
"labeld_WebAttack-XSS.csv",
"labeld_WebAttack-BruteForce.csv",
"labeld_DoS-Slowloris.csv"]

for path, i in zip(unknown, range(5, 13)):
    data = pd.read_csv(f"Data/flow_labeled/{path}")
    data.drop(data.columns[0], inplace=True, axis=1)
    data.insert(len(data.columns), column="label", value=[i]*data.shape[0])

    print(path)
    print(train.shape)
    print(test.shape)

    data.to_csv("Data/test_unknown", mode="a+", header=False, index=False)
