import numpy as np
import pandas as pd
from statistics import mean, stdev
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from math import sqrt
from matplotlib.legend import _get_legend_handles_labels


def metrics_algorithms_multiclass():
    names = [
        "BENIGN",
        "Bot",
        "DDoS",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS slowloris",
        "FTP-Patator",
        "Heartbleed",
        "Infiltration",
        "PortScan",
        "SSH-Patator",
        "Web Attack",
    ]
    with open("5-fold_sets/Results/Algorithms_comparison.txt", "w+") as file:
        macro_average = [[], [], [], [], [], [], [], []]
        weighted_average = [[], [], [], [], [], [], [], []]
        f1_scores = [[], [], [], [], [], [], [], []]
        f1_scores_std = [[], [], [], [], [], [], [], []]
        for idx, alg in enumerate(["EFC", "DT", "SVC", "MLP"]):
            f1 = []
            for sets in range(1, 6):
                y_true = pd.read_csv(
                    "5-fold_sets/Discretized/Sets{}/y_test".format(sets), header=None
                ).values.astype("int")

                y_pred = np.load(
                    "5-fold_sets/Results/Sets{}/{}_predicted.npy".format(sets, alg),
                    allow_pickle=True,
                )
                if alg == "EFC":
                    y_pred = y_pred[0, :].astype("int")

                macro_average[idx].append(f1_score(y_true, y_pred, average="macro"))
                weighted_average[idx].append(
                    f1_score(y_true, y_pred, average="weighted")
                )
                f1.append(
                    f1_score(
                        y_true,
                        y_pred,
                        average=None,
                        labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                    )
                )

            for label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
                lista = []
                for x in range(5):
                    lista.append(f1[x][label])
                f1_scores[idx].append(mean(lista))
                f1_scores_std[idx].append(stdev(lista) / sqrt(len(lista)) * 1.96)

        print(macro_average)
        for label in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            file.write("\\textit{{{}}} ".format(names[label]))
            for idx, alg in enumerate(["DT", "SVC", "MLP", "EFC"]):
                file.write(
                    "& {:.3f} $\\pm$ {:.3f}".format(
                        f1_scores[idx][label], f1_scores_std[idx][label]
                    )
                )
            file.write("\\\\ \n")

        file.write("\\textbf{{{}}} ".format("Macro average"))
        for idx, alg in enumerate(["DT", "SVC", "MLP", "EFC"]):
            file.write(
                "& {:.3f} $\\pm$ {:.3f}".format(
                    mean(macro_average[idx]),
                    (stdev(macro_average[idx]) / sqrt(len(macro_average[idx]))) * 1.96,
                )
            )
        file.write("\\\\ \n")

        file.write("\\textbf{{{}}} ".format("Weighted average"))
        for idx, alg in enumerate(["DT", "SVC", "MLP", "EFC"]):
            file.write(
                "& {:.3f} $\\pm$ {:.3f}".format(
                    mean(weighted_average[idx]),
                    (stdev(weighted_average[idx]) / sqrt(len(weighted_average[idx])))
                    * 1.96,
                )
            )
        file.write("\\\\ \n")


def plot_unknown():
    names = [
        "Bot",
        "DDoS",
        "DoS GoldenEye",
        "DoS Hulk",
        "DoS Slowhttptest",
        "DoS slowloris",
        "FTP-Patator",
        "Heartbleed",
        "Infiltration",
        "PortScan",
        "SSH-Patator",
        "Web Attack",
    ]
    plt.rcParams.update({"font.size": 16})

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex=True, sharey=True, figsize=(15, 9)
    )
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    plt.tight_layout(w_pad=3.3)
    plt.ylim((0.0, 1.4))
    width = 0.45
    for ax, alg in zip([ax1, ax2, ax3, ax4], ["EFC", "DT", "SVC", "MLP"]):
        removed_metrics = []
        for removed in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            normal_percent = []
            others_percent = []
            suspicious_percent = []
            for sets in range(1, 6):
                y_true = np.array(
                    pd.read_csv(
                        "5-fold_sets/Discretized/Sets{}/y_test".format(sets),
                        header=None,
                        squeeze=True,
                    )
                )
                y_pred = list(
                    np.load(
                        "5-fold_sets/Results_removing{}/Sets{}/{}_predicted.npy".format(
                            removed, sets, alg
                        ),
                        allow_pickle=True,
                    )
                )
                if len(y_pred) == 2:
                    y_pred = y_pred[0]

                unknown_predicted = [y_pred[i] for i in np.where(y_true == removed)[0]]
                print(len(unknown_predicted))
                unique, counts = np.unique(unknown_predicted, return_counts=True)

                normal_predicted = counts[np.where(unique == 0)[0]]
                if not normal_predicted:
                    normal_predicted = 0

                suspicious_predicted = counts[np.where(unique == 100)[0]]
                if not suspicious_predicted:
                    suspicious_predicted = 0

                normal_percent.append(float(normal_predicted / len(unknown_predicted)))
                others_percent.append(
                    float(
                        (
                            len(unknown_predicted)
                            - normal_predicted
                            - suspicious_predicted
                        )
                        / len(unknown_predicted)
                    )
                )
                suspicious_percent.append(
                    float(suspicious_predicted / len(unknown_predicted))
                )

            if alg != "EFC":
                removed_metrics.append(
                    [
                        mean(normal_percent),
                        (stdev(normal_percent) / sqrt(len(normal_percent))) * 1.96,
                        mean(others_percent),
                        (stdev(others_percent) / sqrt(len(others_percent))) * 1.96,
                    ]
                )
            else:
                removed_metrics.append(
                    [
                        mean(normal_percent),
                        (stdev(normal_percent) / sqrt(len(normal_percent))) * 1.96,
                        mean(others_percent),
                        (stdev(others_percent) / sqrt(len(others_percent))) * 1.96,
                        mean(suspicious_percent),
                        (stdev(suspicious_percent) / sqrt(len(suspicious_percent)))
                        * 1.96,
                    ]
                )

        if alg != "EFC":
            ax.grid(axis="y", color="gray", linestyle="-", linewidth=0.3)
            ax.bar(
                names,
                [x[0] for x in removed_metrics],
                width,
                ecolor="black",
                yerr=[x[1] for x in removed_metrics],
                error_kw={"elinewidth": 0.5},
                color="#006BA4",
            )
            ax.bar(
                names,
                [x[2] for x in removed_metrics],
                width,
                ecolor="black",
                yerr=[x[3] for x in removed_metrics],
                error_kw={"elinewidth": 0.5},
                bottom=[x[0] for x in removed_metrics],
                color="#CFCFCF",
            )
            ax.set_ylabel("Proportion of predicted samples")
            ticks_loc = ax.get_xticks()
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylim((0.0, 1.4))
            ax.set_axisbelow(True)

            if alg == "SVC":
                ax.set_title(f"SVM")
            else:
                ax.set_title(f"{alg}")

        else:
            ax.grid(axis="y", color="gray", linestyle="-", linewidth=0.3)
            ax.bar(
                names,
                [x[0] for x in removed_metrics],
                width,
                ecolor="black",
                yerr=[x[1] for x in removed_metrics],
                error_kw={"elinewidth": 0.5},
                label="Benign",
                color="#006BA4",
            )
            ax.bar(
                names,
                [x[2] for x in removed_metrics],
                width,
                ecolor="black",
                yerr=[x[3] for x in removed_metrics],
                error_kw={"elinewidth": 0.5},
                bottom=[x[0] for x in removed_metrics],
                label="Other classes",
                color="#CFCFCF",
            )
            ax.bar(
                names,
                [x[4] for x in removed_metrics],
                width,
                ecolor="black",
                yerr=[x[5] for x in removed_metrics],
                error_kw={"elinewidth": 0.5},
                bottom=[x[0] + x[2] for x in removed_metrics],
                label="Suspicious",
                color="#FF800E",
            )
            ax.set_ylabel("Proportion of predicted samples")
            ax.set_xticklabels(names, rotation=45, ha="right")
            ax.set_ylim((0.0, 1.4))
            ax.set_title(f"{alg}")
            ax.set_axisbelow(True)

        fig.legend(*_get_legend_handles_labels(fig.axes))
        fig.savefig(
            "5-fold_sets/Results/4_algs_unknown_CICIDS17.pdf",
            format="pdf",
            bbox_inches="tight",
        )


# metrics_algorithms_multiclass()
plot_unknown()
