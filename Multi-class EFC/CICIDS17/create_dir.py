# import os

# os.makedirs("5-fold_sets/", exist_ok=True)

# os.makedirs("5-fold_sets/Discretized", exist_ok=True)
# os.makedirs("5-fold_sets/Raw", exist_ok=True)
# os.makedirs("5-fold_sets/Normalized", exist_ok=True)
# os.makedirs("5-fold_sets/Encoded", exist_ok=True)
# os.makedirs("5-fold_sets/Results", exist_ok=True)

# for sets in range(1,6):
#     os.makedirs("5-fold_sets/Discretized/Sets{}".format(sets), exist_ok=True)
#     os.makedirs("5-fold_sets/Raw/Sets{}".format(sets), exist_ok=True)
#     os.makedirs("5-fold_sets/Encoded/Sets{}".format(sets), exist_ok=True)
#     os.makedirs("5-fold_sets/Normalized/Sets{}".format(sets), exist_ok=True)
#     os.makedirs("5-fold_sets/Results/Sets{}".format(sets), exist_ok=True)

# for removed in range(1,13):
#     os.makedirs("5-fold_sets/Results_removing{}".format(removed), exist_ok=True)
#     for sets in range(1,6):
#         os.makedirs("5-fold_sets/Results_removing{}/Sets{}".format(removed, sets), exist_ok=True)
import pandas as pd

data = pd.read_csv("TimeData/Normalized/Size4/y_test", header=None)
print(data.shape)

data = pd.read_csv("TimeData/Normalized/Size4/y_train", header=None)
print(data.shape)