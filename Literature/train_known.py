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
pseudo = 0.9
cutoff_p = 0.90

for folder in ["Data", "Data_original"]:
    train = pd.read_csv(f"{folder}/train_known_binary.csv", header=None)

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
        train[:, :-1], train[:, -1], n_bins, pseudo, cutoff_p
    )

    np.save(f"{folder}/Models/h_i_09_90.npy", h_i_matrices)
    np.save(f"{folder}/Models/couplings_09_90.npy", coupling_matrices)
    np.save(f"{folder}/Models/cutoffs_09_90.npy", cutoffs_list)
    np.save(f"{folder}/Models/train_labels.npy", train[:, -1])
    pickle.dump(preprocessor, open(f"{folder}/Models/preprocessor.pkl", mode="wb"))
