from statistics import mean
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_curve, roc_curve, auc


def single_attack(folder, attacks):
    aucs = [[], [], []]
    with open(f"{folder}/1by1_aurocs.txt", mode="w+") as file:
        for attack in attacks:
            y_test = pd.read_csv(
                f"{folder}/1by1_sets/{attack}.csv", header=None
            ).values[:, -1]
            y_score = np.load(
                f"{folder}/Results/1by1/EFC_{attack}.npy", allow_pickle=True
            )[1]
            probs_rvs = [
                x for x in MinMaxScaler().fit_transform(y_score.reshape(-1, 1))
            ]

            probs = [1 - x for x in probs_rvs]

            fpr, tpr, _ = roc_curve(y_test, probs, pos_label=1)
            precision, recall, _ = precision_recall_curve(y_test, probs, pos_label=1)
            precision_rvs, recall_rvs, _ = precision_recall_curve(
                y_test, probs_rvs, pos_label=1
            )

            aucs[0].append(auc(fpr, tpr))
            aucs[1].append(auc(recall, precision))
            aucs[2].append(auc(recall_rvs, precision_rvs))

            file.write(
                "{} & {:.3f} & {:.3f} & {:.3f}\n".format(
                    attack,
                    auc(fpr, tpr),
                    auc(recall, precision),
                    auc(recall_rvs, precision_rvs),
                )
            )
        file.write(
            "Avg & {:.3f} & {:.3f} & {:.3f}\n".format(
                mean(aucs[0]),
                mean(aucs[1]),
                mean(aucs[2]),
            )
        )


attacks = [
    "DoS-Slowloris",
    "WebAttacks",
    "Heartbleed-Port",
    "DoS-Slowhttptest",
    "Infiltration",
    "Botnet",
]


single_attack("Data", attacks)
