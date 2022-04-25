import pandas as pd
import os
import resource

sizes = [1,2,3,4]
os.makedirs("TimeData/", exist_ok=True)
os.makedirs("TimeData/Discretized", exist_ok=True)
os.makedirs("TimeData/Normalized", exist_ok=True)
os.makedirs("TimeData/Results", exist_ok=True)
for i in sizes:
    os.makedirs("TimeData/Discretized/Size{}".format(i), exist_ok=True)
    os.makedirs("TimeData/Normalized/Size{}".format(i), exist_ok=True)
    os.makedirs("TimeData/Results/Size{}".format(i), exist_ok=True)

def main():
    for i in range(1,len(sizes)+1):
        X_test = pd.concat( [ pd.read_csv(f"5-fold_sets/Discretized/Sets{j}/X_test", header=None) for j in sizes[:i] ] )
        y_test = pd.concat( [ pd.read_csv(f"5-fold_sets/Discretized/Sets{j}/y_test", header=None) for j in sizes[:i] ] )

        X_train = pd.concat( [ pd.read_csv(f"5-fold_sets/Discretized/Sets{j}/X_train", header=None) for j in sizes[:i] ] )
        y_train = pd.concat( [ pd.read_csv(f"5-fold_sets/Discretized/Sets{j}/y_train", header=None) for j in sizes[:i] ] )

        X_test.to_csv(f"TimeData/Discretized/Size{i}/X_test", header=False, index=False)
        y_test.to_csv(f"TimeData/Discretized/Size{i}/y_test", header=False, index=False)
        X_train.to_csv(f"TimeData/Discretized/Size{i}/X_train", header=False, index=False)
        y_train.to_csv(f"TimeData/Discretized/Size{i}/y_train", header=False, index=False)

        X_test = pd.concat( [ pd.read_csv(f"5-fold_sets/Normalized/Sets{j}/X_test", header=None) for j in sizes[:i] ] )
        y_test = pd.concat( [ pd.read_csv(f"5-fold_sets/Normalized/Sets{j}/y_test", header=None) for j in sizes[:i] ] )

        X_train = pd.concat( [ pd.read_csv(f"5-fold_sets/Normalized/Sets{j}/X_train", header=None) for j in sizes[:i] ] )
        y_train = pd.concat( [ pd.read_csv(f"5-fold_sets/Normalized/Sets{j}/y_train", header=None) for j in sizes[:i] ] )

        X_test.to_csv(f"TimeData/Normalized/Size{i}/X_test", header=False, index=False)
        y_test.to_csv(f"TimeData/Normalized/Size{i}/y_test", header=False, index=False)
        X_train.to_csv(f"TimeData/Normalized/Size{i}/X_train", header=False, index=False)
        y_train.to_csv(f"TimeData/Normalized/Size{i}/y_train", header=False, index=False)


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
