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
        cutoff = define_cutoff(data, h_i, couplingmatrix, n_bins)
        coupling_matrices.append(couplingmatrix)
        h_i_matrices.append(h_i)
        cutoffs_list.append(cutoff)

    return h_i_matrices, coupling_matrices, cutoffs_list

def test_multiclass_model(test_data, h_i_matrices, coupling_matrices, cutoffs_list, n_bins, train_labels):
    predicted = []
    energies_results = []
    unknown = 0
    for i in range(test_data.shape[0]):
        energies = []
        for indx in range(len(h_i_matrices)):
            e = 0
            for j in range(test_data.shape[1]-1):
                for k in range(j,test_data.shape[1]):
                    if test_data[i,j] != (n_bins-1) and test_data[i,k] != (n_bins-1):
                        e -= np.log(coupling_matrices[indx][int(j*(n_bins-1) + test_data[i,j]), int(k*(n_bins-1) + test_data[i,k])])
                if test_data[i,j] != (n_bins-1):
                    e -= np.log(h_i_matrices[indx][int(j*(n_bins-1) + test_data[i,j])])
            energies.append(e)
        min_energie = min(energies)
        indx = energies.index(min_energie)
        if min_energie < cutoffs_list[indx]:
            predicted.append(train_labels[indx])
        else:
            predicted.append(100)
            unknown += 1

    return predicted, unknown
