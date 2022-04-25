import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import sys
sys.path.append("../../../EFC")
from classification_functions import *
import os
import time
import resource


def DT(sets):
    train = pd.read_csv("TimeData/Normalized/Size{}/X_train".format(sets), header=None)
    test = pd.read_csv("TimeData/Normalized/Size{}/X_test".format(sets), header=None)
    train_labels = pd.read_csv("TimeData/Normalized/Size{}/y_train".format(sets), header=None).squeeze()

    DT = DecisionTreeClassifier()
    start = time.time()
    DT.fit(train, train_labels)
    training_time = time.time()-start


    start = time.time()
    predicted = DT.predict(test)
    testing_time = time.time()-start
    print("DT train:", training_time)
    print("DT test:", testing_time)
    np.save("TimeData/Results/Size{}/DT_times.npy".format(sets), [training_time, testing_time])


def svc(sets):
    train = pd.read_csv("TimeData/Normalized/Size{}/X_train".format(sets), header=None)
    test = pd.read_csv("TimeData/Normalized/Size{}/X_test".format(sets), header=None)
    train_labels = pd.read_csv("TimeData/Normalized/Size{}/y_train".format(sets), header=None).squeeze()


    svc = SVC(kernel='poly', probability=True)
    start = time.time()
    svc.fit(train, train_labels)
    training_time = time.time()-start

    start = time.time()
    predicted = svc.predict(test)
    testing_time = time.time()-start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save("TimeData/Results/Size{}/SVC_times.npy".format(sets), [training_time, testing_time])

def mlp(sets):
    train = pd.read_csv("TimeData/Normalized/Size{}/X_train".format(sets), header=None)
    test = pd.read_csv("TimeData/Normalized/Size{}/X_test".format(sets), header=None)
    train_labels = pd.read_csv("TimeData/Normalized/Size{}/y_train".format(sets), header=None).squeeze()

    MLP = MLPClassifier(max_iter=300)
    start = time.time()
    MLP.fit(train, train_labels)
    training_time = time.time()-start

    start = time.time()
    predicted = MLP.predict(test)
    testing_time = time.time()-start
    print("Train:", training_time)
    print("Test:", testing_time)
    np.save("TimeData/Results/Size{}/MLP_times.npy".format(sets), [training_time, testing_time])

def EFC(sets):
    train = pd.read_csv("TimeData/Discretized/Size{}/X_train".format(sets), header=None).astype('int')
    train_labels =  pd.read_csv("TimeData/Discretized/Size{}/y_train".format(sets), header=None).astype('int').squeeze()
    test = pd.read_csv("TimeData/Discretized/Size{}/X_test".format(sets), header=None).astype('int')
    test_labels = pd.read_csv("TimeData/Discretized/Size{}/y_test".format(sets), header=None).astype('int').squeeze()

    Q = 30
    LAMBDA = 0.5

    start = time.time()
    h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(
        np.array(train), np.array(train_labels), Q, LAMBDA
    )
    training_time = time.time() - start
    print("EFC train: ", training_time)

    start = time.time()
    predicted, energies = MultiClassPredict(
        np.array(test),
        h_i_matrices,
        coupling_matrices,
        cutoffs_list,
        Q,
        np.unique(train_labels),
    )
    testing_time = time.time() - start
    print("EFC test: ", testing_time)
    np.save("TimeData/Results/Size{}/EFC_times.npy".format(sets), [training_time, testing_time])

    print("Train:", training_time)
    print("Test:", testing_time)

def main():
    for size in [1,2,3,4]:
        os.makedirs("TimeData/Results/Size{}/".format(size), exist_ok=True)
        EFC(size)
        DT(size)
        svc(size)
        mlp(size)


def memory_limit():
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    resource.setrlimit(resource.RLIMIT_AS, (get_memory() * 1024 * 0.8, hard))


def get_memory():
    with open("/proc/meminfo", "r") as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ("MemFree:", "Buffers:", "Cached:"):
                free_memory += int(sline[1])
    return free_memory


if __name__ == "__main__":
    memory_limit()  # Limitates maximun memory usage to half
    try:
        main()
    except MemoryError:
        print("Memory error")
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)
