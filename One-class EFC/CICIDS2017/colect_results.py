import numpy as np
import pandas as pd
from statistics import stdev, mean

#this script creates the results tables shown in the paper
#these tables includes the mean result of the cross validation and the result fo the external test

files = ['Friday-WorkingHours-Afternoon-DDos.pcap_ISCX','Wednesday-workingHours.pcap_ISCX']
algorithms = ['GaussianNB','KNN', 'DT', 'Adaboost', 'RF', 'SVC', 'ANN','EFC']

for file in files:
    results = open("Cross_validation/Results/{}/Mean_folds_and_external_results.txt".format(file), 'w+')
    for j in algorithms:
        precision_intern = []
        recall_intern = []
        f1_intern = []
        roc_intern = []
        precision_extern = []
        recall_extern = []
        f1_extern = []
        roc_extern = []
        for i in range(1,11):
            temp_intern = np.load("Cross_validation/Results/{}/Sets{}/{}.npy".format(file,i, j))
            temp_extern = np.load("External_test/Results/Training_{}/Sets{}/{}.npy".format(file,i, j))

            precision_intern.append(temp_intern[0])
            recall_intern.append(temp_intern[1])
            f1_intern.append(temp_intern[2])
            roc_intern.append(temp_intern[3])

            precision_extern.append(temp_extern[0])
            recall_extern.append(temp_extern[1])
            f1_extern.append(temp_extern[2])
            roc_extern.append(temp_extern[3])
        results.write('{} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} & {:.3f} $\\pm$ {:.3f} \\\\ \n'.format(j, mean(f1_intern), stdev(f1_intern), mean(roc_intern), stdev(roc_intern), mean(f1_extern), stdev(f1_extern), mean(roc_extern), stdev(roc_extern)))
    results.close()
