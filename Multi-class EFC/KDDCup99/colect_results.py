import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, balanced_accuracy_score
import seaborn as sns


# predicted = np.load("Results/EFC_predicted.npy", allow_pickle=True)
# test_labels =  np.array(pd.read_csv("Data/Normalized-Discretized/y_test", squeeze=True, header=None).astype('int'))
#
# plt.rcParams.update({'font.size': 14})
# names = ['Normal','DoS','Probe', 'R2L', 'U2R', 'Unknown']
#
# cf = pd.DataFrame(confusion_matrix(test_labels, predicted, normalize='true'), index = names, columns = names)
# sns.heatmap(cf, annot=True, cmap="Blues", fmt='.2f')
# plt.ylabel("True label")
# plt.xlabel("Predicted label")
# plt.yticks(rotation=30)
# plt.xticks(rotation=30)
# plt.tight_layout()
# plt.savefig("CM_Test.pdf", format='pdf',bbox_inches = "tight")
# plt.show()

def plot_unknown():
    algorithms = ['NB','KNN', 'DT','SVC', 'MLP','RF','EFC']
    benign = []
    other_classes = []
    unknown = []
    unknown_idx = list(np.load("unknown_idx.npy", allow_pickle=True))
    total = len(unknown_idx)
    for alg in algorithms:
        predicted = np.load("Results/{}_predicted.npy".format(alg), allow_pickle=True)
        other_classes.append(np.where(np.isin([predicted[k] for k in unknown_idx], [1,2,3,4]))[0].shape[0]/total)
        benign.append(np.where(np.isin([predicted[k] for k in unknown_idx], [0]))[0].shape[0]/total)
        if alg == 'EFC':
            unknown.append(np.where(np.isin([predicted[k] for k in unknown_idx], [100]))[0].shape[0]/total)
        else:
            unknown.append(0)

    print(other_classes)
    print(benign)
    print(unknown)
    # plt.tight_layout(w_pad=3.3)
    plt.ylim(0,1.2)
    width = 0.45
    plt.bar(algorithms, benign, width, capsize=3, label='Benign', color='#006BA4')
    plt.bar(algorithms, other_classes, width, capsize=3, bottom=benign, label='Other classes',  color = '#CFCFCF')
    plt.bar(algorithms, unknown, width, capsize=3, bottom=[x+y for x,y in zip(benign, other_classes)], label='Suspicious',  color = '#FF800E')
    plt.title('Classification of unknown samples')
    plt.ylabel('Percentage of predicted samples')
    plt.legend(loc=1, bbox_to_anchor=(1.1, 1.05))
    plt.savefig("new_attacks_classification.pdf", fmt="pdf")
    plt.show()

def metrics_algorithms_multiclass():
    names = ['Normal','DoS','Probe', 'R2L', 'U2R', 'Unknown']
    with open("Results/Algorithms_comparison.txt", 'w+') as file:
        macro_avarege = []
        balanced_acc = []
        f1_scores = []
        for idx, alg in enumerate(['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']):
            y_pred = np.load("Results/{}_predicted.npy".format(alg), allow_pickle=True)
            y_true =  np.array(pd.read_csv("Data/Normalized-Discretized/y_test", squeeze=True, header=None).astype('int'))
            macro_avarege.append(f1_score(y_true, y_pred, average='macro'))
            balanced_acc.append(balanced_accuracy_score(y_true, y_pred))
            f1_scores.append(f1_score(y_true, y_pred, average=None, labels=[0, 1, 2, 3, 4]))


        for label in [0, 1, 2, 3, 4]:
            file.write('\\textit{{{}}} '.format(names[label]))
            for idx, alg in enumerate(['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']):
                print(label, idx)
                file.write('& {:.3f} '.format(f1_scores[idx][label]))
            file.write('\\\\ \n')

        file.write('\\textbf{{{}}} '.format('Macro average'))
        for idx, alg in enumerate(['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']):
                file.write('& {:.3f} '.format(macro_avarege[idx]))
        file.write('\\\\ \n')

        file.write('\\textbf{{{}}} '.format('Balanced accuracy'))
        for idx, alg in enumerate(['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']):
                file.write('& {:.3f} '.format(balanced_acc[idx]))
        file.write('\\\\ \n')

        unknown_idx = list(np.load("unknown_idx.npy", allow_pickle=True))
        total = len(unknown_idx)
        file.write('\\textbf{{{}}} '.format('Unknown'))
        for alg in ['EFC','NB','KNN', 'DT', 'SVC', 'MLP','RF']:
            predicted = np.load("Results/{}_predicted.npy".format(alg), allow_pickle=True)
            other = np.where(np.isin([predicted[k] for k in unknown_idx], [1,2,3,4]))[0].shape[0]/total
            unknown = 0
            if alg == 'EFC':
                unknown = np.where(np.isin([predicted[k] for k in unknown_idx], [100]))[0].shape[0]/total
            file.write('& {:.3f} '.format(other+unknown))


metrics_algorithms_multiclass()
plot_unknown()
