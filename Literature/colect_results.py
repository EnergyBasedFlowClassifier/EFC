from statistics import mean
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, accuracy_score, roc_curve, auc
from torch import threshold


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


def open_setness():
    with open("N_unknown_accs.txt", mode="w+") as file:
        for folder in ["Data", "Data_original"]:
            for i in range(2, 5):
                y_test = (
                    pd.read_csv(f"{folder}/N_unknown_sets/N_{i}.csv")
                    .values[:, -1]
                    .astype("int")
                )
                y_pred, y_score = np.load(
                    f"{folder}/Results/N_unknown/EFC_N_{i}.npy", allow_pickle=True
                )

                y_pred = y_pred.astype("int")

                # y_score = np.array([x[0] for x in y_score])

                # cutoff = np.load(
                #     f"{folder}/Models/cutoffs_09_90.npy", allow_pickle=True
                # )[0]
                # color_attack = "#006680"
                # color_normal = "#b3b3b3"

                # known = y_score[np.where(y_test == 1)[0]]
                # unknown = y_score[np.where(y_test == 0)[0]]
                # print(len(known), len(unknown))
                # bins = np.histogram(np.hstack((known, unknown)), bins=30)[1]

                # plt.hist(
                #     unknown,
                #     bins,
                #     facecolor=color_attack,
                #     alpha=0.7,
                #     ec="white",
                #     linewidth=0.3,
                #     label="unknown",
                # )
                # plt.hist(
                #     known,
                #     bins,
                #     facecolor=color_normal,
                #     alpha=0.7,
                #     ec="white",
                #     linewidth=0.3,
                #     label="known",
                # )
                # plt.axvline(cutoff)
                # plt.savefig(f"plots_{i}.png", format="png")
                # plt.close()
                unknown = np.where(y_test == 0)[0].astype("int")

                file.write(
                    "{:.3f} & {:.3f}  & & ".format(
                        accuracy_score(y_test, y_pred),
                        accuracy_score(y_test[unknown], y_pred[unknown]),
                    )
                )
            file.write("\n")


attacks = [
    "DoS-Slowloris",
    "WebAttacks",
    "Heartbleed-Port",
    "DoS-Slowhttptest",
    "Infiltration",
    "Botnet",
]

unknown_original = [
    "DoS slowloris",
    "Heartbleed",
    "Infiltration",
    "Bot",
    "DoS Slowhttptest",
    "Web Attack",
]


# single_attack("Data", attacks)
# single_attack("Data_original", unknown_original)
open_setness()
