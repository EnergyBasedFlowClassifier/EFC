import pandas as pd
import sys

sys.path.append("../../EFC")
from classification_functions import *
import resource
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import itertools


def predict(args):
    i, folder = args
    n_bins = 30
    test = pd.read_csv(f"{folder}/N_unknown_sets/N_{i}.csv").values
    train_labels = np.load(f"{folder}/Models/train_labels.npy", allow_pickle=True)
    preprocessor = pickle.load(open(f"{folder}/Models/preprocessor.pkl", mode="rb"))
    test = preprocessor.transform(test).astype("int64")
    print(test.shape)

    h_i = np.load(f"{folder}/Models/h_i_09_90.npy", allow_pickle=True)
    couplings = np.load(f"{folder}/Models/couplings_09_90.npy", allow_pickle=True)
    cutoffs = np.load(f"{folder}/Models/cutoffs_09_90.npy", allow_pickle=True)

    y_pred, y_score = MultiClassPredict(
        np.array(test[:, :-1]), h_i, couplings, cutoffs, n_bins, np.unique(train_labels)
    )

    np.save(f"{folder}/Results/N_unknown/EFC_N_{i}.npy", [y_pred, y_score])


def main():
    for arg in itertools.product(range(1, 7), ["Data", "Data_original"]):
        predict(arg)


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
        sys.stderr.write("\n\nERROR: Memory Exception\n")
        sys.exit(1)
