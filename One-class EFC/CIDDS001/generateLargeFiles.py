import numpy as np
import os

os.makedirs("TimeData/", exist_ok=True)
os.makedirs("TimeData/Discretized", exist_ok=True)
os.makedirs("TimeData/Non_discretized", exist_ok=True)
os.makedirs("TimeData/Results", exist_ok=True)
for i in [0,1,2,3]:
    os.makedirs("TimeData/Discretized/Size{}".format(i), exist_ok=True)
    os.makedirs("TimeData/Non_discretized/Size{}".format(i), exist_ok=True)
    os.makedirs("TimeData/Results/Size{}".format(i), exist_ok=True)

test_d = np.load("Data/Discretized/Exp1/test.npy", allow_pickle=True)
test_labels_d = np.load("Data/Discretized/Exp1/test_labels.npy", allow_pickle=True)
train_d = np.load("Data/Discretized/Exp1/train.npy", allow_pickle=True)
train_labels_d = np.load("Data/Discretized/Exp1/train_labels.npy", allow_pickle=True)

test_nd = np.load("Data/Non_discretized/Exp1/test.npy", allow_pickle=True)
test_labels_nd = np.load("Data/Non_discretized/Exp1/test_labels.npy", allow_pickle=True)
train_nd = np.load("Data/Non_discretized/Exp1/train.npy", allow_pickle=True)
train_labels_nd = np.load("Data/Non_discretized/Exp1/train_labels.npy", allow_pickle=True)

np.save("TimeData/Discretized/Size0/test.npy", test_d)
np.save("TimeData/Discretized/Size0/test_labels.npy", test_labels_d)
np.save("TimeData/Discretized/Size0/train.npy", train_d)
np.save("TimeData/Discretized/Size0/train_labels.npy", train_labels_d)

np.save("TimeData/Non_discretized/Size0/test.npy", test_nd)
np.save("TimeData/Non_discretized/Size0/test_labels.npy", test_labels_nd)
np.save("TimeData/Non_discretized/Size0/train.npy", train_nd)
np.save("TimeData/Non_discretized/Size0/train_labels.npy", train_labels_nd)


for turn in range(1,11):
    test_d = np.concatenate((np.load("Data/Discretized/Exp1/test.npy", allow_pickle=True), test_d), axis=0)
    test_labels_d = np.concatenate((np.load("Data/Discretized/Exp1/test_labels.npy", allow_pickle=True), test_labels_d), axis=0)
    train_d = np.concatenate((np.load("Data/Discretized/Exp1/train.npy", allow_pickle=True), train_d), axis=0)
    train_labels_d = np.concatenate((np.load("Data/Discretized/Exp1/train_labels.npy", allow_pickle=True), train_labels_d), axis=0)

    test_nd = np.concatenate((np.load("Data/Non_discretized/Exp1/test.npy", allow_pickle=True), test_nd), axis=0)
    test_labels_nd = np.concatenate((np.load("Data/Non_discretized/Exp1/test_labels.npy", allow_pickle=True), test_labels_nd), axis=0)
    train_nd = np.concatenate((np.load("Data/Non_discretized/Exp1/train.npy", allow_pickle=True), train_nd), axis=0)
    train_labels_nd = np.concatenate((np.load("Data/Non_discretized/Exp1/train_labels.npy", allow_pickle=True), train_labels_nd), axis=0)

    if turn in [1,2,3]:
        np.save("TimeData/Discretized/Size{}/test.npy".format(turn), test_d)
        np.save("TimeData/Discretized/Size{}/test_labels.npy".format(turn), test_labels_d)
        np.save("TimeData/Discretized/Size{}/train.npy".format(turn), train_d)
        np.save("TimeData/Discretized/Size{}/train_labels.npy".format(turn), train_labels_d)

        np.save("TimeData/Non_discretized/Size{}/test.npy".format(turn), test_nd)
        np.save("TimeData/Non_discretized/Size{}/test_labels.npy".format(turn), test_labels_nd)
        np.save("TimeData/Non_discretized/Size{}/train.npy".format(turn), train_nd)
        np.save("TimeData/Non_discretized/Size{}/train_labels.npy".format(turn), train_labels_nd)
