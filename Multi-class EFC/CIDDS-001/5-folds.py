import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

def cv_discretized():
    train_id = [[],[],[],[],[]]
    test_id = [[],[],[],[],[]]

    reader = pd.read_csv("CIDDS-001/traffic/OpenStack/Discretized.csv", chunksize=4000000, header=None)
    for chunk in reader:
        skf = StratifiedKFold(n_splits=5)

        X = chunk.iloc[:, 0:-1]
        y = chunk.iloc[:, -1]

        count = 1

        for train_index, test_index in skf.split(X, y):
            print(count)
            X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            X_train.to_csv("5-fold_sets/Discretized/Sets{}/train.csv".format(count), mode='a', header=False, index=False)
            X_test.to_csv("5-fold_sets/Discretized/Sets{}/test.csv".format(count), mode='a', header=False, index=False)
            y_train.to_csv("5-fold_sets/Discretized/Sets{}/train_labels.csv".format(count), mode='a', header=False, index=False)
            y_test.to_csv("5-fold_sets/Discretized/Sets{}/test_labels.csv".format(count), mode='a', header=False, index=False)

            train_id[count-1].append(train_index)
            test_id[count-1].append(test_index)
            count+=1

    np.save("train_id.npy", train_id)
    np.save("test_id.npy", test_id)


def cv_non_discretized():
    train_id = np.load("train_id.npy", allow_pickle=True)
    test_id = np.load("test_id.npy", allow_pickle=True)

    reader_pre = pd.read_csv("CIDDS-001/traffic/OpenStack/Pre_processed_encoded.csv", chunksize=4000000, header=None)
    n_chunk = 0
    for chunk in reader_pre:
        X = chunk.iloc[:, 0:-1]
        y = chunk.iloc[:, -1]

        for count in range(1,6):
            print(count)
            X_train, X_test = X.iloc[train_id[count-1][n_chunk], :], X.iloc[test_id[count-1][n_chunk], :]
            y_train, y_test = y.iloc[train_id[count-1][n_chunk]], y.iloc[test_id[count-1][n_chunk]]


            X_train.to_csv("5-fold_sets/Non_discretized/Sets{}/train.csv".format(count), mode='a', header=False, index=False)
            X_test.to_csv("5-fold_sets/Non_discretized/Sets{}/test.csv".format(count), mode='a', header=False, index=False)
            y_train.to_csv("5-fold_sets/Non_discretized/Sets{}/train_labels.csv".format(count), mode='a', header=False, index=False)
            y_test.to_csv("5-fold_sets/Non_discretized/Sets{}/test_labels.csv".format(count), mode='a', header=False, index=False)
        n_chunk += 1

cv_discretized()
cv_non_discretized()
