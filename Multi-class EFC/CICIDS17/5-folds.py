import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

reader = pd.read_csv("TrafficLabelling /Pre_processed_unique.csv", chunksize=2_000_000, header=None)
for chunk in reader:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    X = chunk.iloc[:, 0:-1]
    y = chunk.iloc[:, -1]

    count = 1

    for train_index, test_index in skf.split(X, y):
        print(count)
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        X_train.to_csv("5-fold_sets/Non_discretized/Sets{}/train.csv".format(count), mode='a', header=False, index=False)
        X_test.to_csv("5-fold_sets/Non_discretized/Sets{}/test.csv".format(count), mode='a', header=False, index=False)
        y_train.to_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(count), mode='a', header=False, index=False)
        y_test.to_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(count), mode='a', header=False, index=False)

        count+=1
