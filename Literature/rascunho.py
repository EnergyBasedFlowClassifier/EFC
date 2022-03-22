import pandas as pd

train = pd.read_csv("Data/train_known", header=None)
test1 = pd.read_csv("Data/test_known", header=None)
test2 = pd.read_csv("Data/test_unknown", header=None)

print(train.shape)
print(test1.shape)
print(test2.shape)
