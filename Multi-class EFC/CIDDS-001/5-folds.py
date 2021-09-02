import pandas as pd
from sklearn.model_selection import StratifiedKFold

data = pd.read_csv("CIDDS-001/traffic/OpenStack/Pre_processed.csv", header=None)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

x = data.iloc[:, :-1]
y = data.iloc[:, -1]

count = 1

for train_index, test_index in skf.split(x, y):
    print(count)
    data.iloc[train_index,:].to_csv("5-fold_sets/Raw/Sets{}/train.csv".format(count), mode='a', header=False, index=False)
    data.iloc[test_index,:].to_csv("5-fold_sets/Raw/Sets{}/test.csv".format(count), mode='a', header=False, index=False)

    count+=1
