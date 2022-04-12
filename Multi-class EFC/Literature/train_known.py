import pandas as pd
import sys

sys.path.append("../../EFC")
from classification_functions import *
import numpy as np
from sklearn.preprocessing import MaxAbsScaler, KBinsDiscretizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle

n_bins = 30
pseudo = 0.5

folder = "Data"

train = pd.read_csv(f"{folder}/train_known.csv", header=None)

numeric_transformer = Pipeline(
    steps=[
        ("scaler", MaxAbsScaler()),
        (
            "discretizer",
            KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile"),
        ),
    ]
)
preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_transformer, train.columns[:-1]),
        ("categorical", "passthrough", [-1]),
    ]
)

train = preprocessor.fit_transform(train).astype("int")
print(np.unique(train[:, -1]))

h_i_matrices, coupling_matrices, cutoffs_list = MultiClassFit(
    train[:, :-1], train[:, -1], n_bins, pseudo
)

np.save(f"{folder}/Models/h_i.npy", h_i_matrices)
np.save(f"{folder}/Models/couplings.npy", coupling_matrices)
np.save(f"{folder}/Models/cutoffs.npy", cutoffs_list)
np.save(f"{folder}/Models/train_labels.npy", train[:, -1])
pickle.dump(preprocessor, open(f"{folder}/Models/preprocessor.pkl", mode="wb"))
