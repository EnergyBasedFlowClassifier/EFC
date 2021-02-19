import numpy as np
from dca_functions import *
import numpy as np
from dca_functions import *

def create_oneclass_model(data,n_bins, lamb):
    sitefreq = Sitefreq(data, n_bins, lamb)
    pairfreq = Pairfreq(data, sitefreq, n_bins, lamb)
    couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
    h_i = local_fields(couplingmatrix, sitefreq, n_bins)
    cutoff = define_cutoff(data, h_i, couplingmatrix, n_bins)
    return h_i, couplingmatrix, cutoff

def test_oneclass_model(test_data, model, h_i, expected, cutoff, n_bins):
    results = np.zeros((6),dtype=int)
    energies = []
    predicted = []
    for i in range(test_data.shape[0]):
        e = 0
        for j in range(test_data.shape[1]-1):
            for k in range(j,test_data.shape[1]):
                if test_data[i,j] != (n_bins-1) and test_data[i,k] != (n_bins-1):
                    e -= np.log(model[int(j*(n_bins-1) + test_data[i,j]), int(k*(n_bins-1) + test_data[i,k])])
            if test_data[i,j] != (n_bins-1):
                e -= np.log(h_i[int(j*(n_bins-1) + test_data[i,j])])
        if (e <= cutoff) and (expected[i] == 0):
            results[0] += 1
            predicted.append(0)
        elif e <= cutoff:
            results[3] += 1
            predicted.append(0)
            print(e, test_data[i,:])
        elif (e > cutoff) and (expected[i] != 0):
            results[1] += 1
            predicted.append(1)
        elif (e > cutoff):
            results[2] += 1
            predicted.append(1)
        energies.append(e)
    return predicted, energies

@profile
def create_multiclass_model(data,labels,n_bins, lamb):
    data_per_type = []
    for label in np.unique(labels):
        selected = [data[i,:] for i in range(data.shape[0]) if labels[i] == label]
        data_per_type.append(np.array(selected))
    h_i_matrices = []
    coupling_matrices = []
    cutoffs_list = []
    for set in data_per_type:
        sitefreq = Sitefreq(set, n_bins, lamb)
        pairfreq = Pairfreq(set, sitefreq, n_bins, lamb)
        couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
        h_i = local_fields(couplingmatrix, sitefreq, n_bins)
        cutoff = define_cutoff(set, h_i, couplingmatrix, n_bins)
        coupling_matrices.append(couplingmatrix)
        h_i_matrices.append(h_i)
        cutoffs_list.append(cutoff)

    return h_i_matrices, coupling_matrices, cutoffs_list

@profile
def test_multiclass_model(test_data, h_i_matrices, coupling_matrices, cutoffs_list, n_bins, train_labels):
    n_rows = test_data.shape[0]
    n_columns = test_data.shape[1]
    n_classes = len(h_i_matrices)
    result = np.empty(n_rows, dtype=int)
    for i in range(n_rows):
        min_energy = None
        row = test_data[i,:]
        for label in range(n_classes):
            e = 0
            couplingmatrix = coupling_matrices[label]
            h_i = h_i_matrices[label]
            for j in range(n_columns-1):
                j_value = int(row[j])
                j_value_position = j*(n_bins-1) + j_value
                for k in range(j,n_columns):
                    k_value = int(row[k])
                    if j_value != (n_bins-1) and k_value != (n_bins-1):
                        e -= np.log(couplingmatrix[j_value_position, k*(n_bins-1)+k_value])
                if j_value != (n_bins-1):
                    e -= np.log(h_i[j_value_position])
            if min_energy is None or e < min_energy:
                min_energy = e
                predicted_label = label

        if min_energy < cutoffs_list[predicted_label]:
            result[i] = predicted_label
        else:
            result[i] = 100
    return result
