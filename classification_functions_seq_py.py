import numpy as np
from dca_functions import *

def DefineCutoff(train_data, h_i, couplingmatrix, Q):
    n_inst = train_data.shape[0]
    n_attr = train_data.shape[1]
    energies = np.empty(n_inst, dtype=np.float64)
    predicted = np.empty(n_inst, dtype=bool)
    for i in range(n_inst):
        e = 0
        for j in range(n_attr-1):
            j_value = train_data[i,j]
            if j_value != (Q-1):
                for k in range(j,n_attr):
                    k_value = train_data[i,k]
                    if k_value != (Q-1):
                        e -= (couplingmatrix[j*(Q-1) + j_value, k*(Q-1) + k_value])
                e -= (h_i[j*(Q-1) + j_value])
        energies[i] = e
    energies = np.sort(energies, axis=None)
    cutoff = energies[int(energies.shape[0]*0.95)]
    return cutoff

def OneClassFit(data, Q, LAMBDA):
    sitefreq = Sitefreq(data, Q, LAMBDA)
    pairfreq = Pairfreq(data, sitefreq, Q, LAMBDA)
    couplingmatrix = Coupling(sitefreq, pairfreq, Q)
    h_i = LocalFields(couplingmatrix, sitefreq, Q)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cutoff = DefineCutoff(data, h_i, couplingmatrix, Q)
    return couplingmatrix, h_i, cutoff

def OneClassPredict(test_data, model, h_i, cutoff, Q):
    n_inst = test_data.shape[0]
    n_attr = test_data.shape[1]
    energies = np.empty(n_inst, dtype=np.float64)
    predicted = np.empty(n_inst, dtype=bool)
    for i in range(n_inst):
        e = 0
        for j in range(n_attr-1):
            j_value = test_data[i,j]
            if j_value != (Q-1):
                for k in range(j,n_attr):
                    k_value = test_data[i,k]
                    if k_value != (Q-1):
                        e -= (model[j*(Q-1) + j_value, k*(Q-1) + k_value])
                e -= (h_i[j*(Q-1) + j_value])
        predicted[i] = e > cutoff
        energies[i] = e
    return predicted, energies


def MultiClassFit(data, labels, Q, LAMBDA):
    data_concat = np.empty((data.shape[0],data.shape[1]+1))
    data_concat[:,:-1] = data
    data_concat[:, -1] = labels
    h_i_matrices = []
    coupling_matrices = []
    cutoffs_list = []
    for label in np.unique(labels):
        subset = data_concat[data_concat[:,-1] == label]
        sitefreq = Sitefreq(subset, Q, LAMBDA)
        pairfreq = Pairfreq(subset, sitefreq, Q, LAMBDA)
        couplingmatrix = Coupling(sitefreq, pairfreq, Q)
        h_i = LocalFields(couplingmatrix, sitefreq, Q)
        couplingmatrix = np.log(couplingmatrix)
        h_i = np.log(h_i)
        cutoff = DefineCutoff(subset, h_i, couplingmatrix, Q)
        coupling_matrices.append(couplingmatrix)
        h_i_matrices.append(h_i)
        cutoffs_list.append(cutoff)

    return h_i_matrices, coupling_matrices, cutoffs_list


def MultiClassPredict(test_data, h_i_matrices, coupling_matrices, cutoffs_list, Q, train_labels):
    n_inst = test_data.shape[0]
    n_attr = test_data.shape[1]
    n_classes = len(h_i_matrices)
    predicted = np.empty(n_inst, dtype=int)
    for i in range(n_inst):
        energies = []
        for label in range(n_classes):
            e = 0
            couplingmatrix = coupling_matrices[label]
            h_i = h_i_matrices[label]
            for j in range(n_attr-1):
                j_value = test_data[i,j]
                if j_value != (Q-1):
                    for k in range(j,n_attr):
                        k_value = test_data[i,k]
                        if k_value != (Q-1):
                            e -= (couplingmatrix[j*(Q-1) + j_value, k*(Q-1) + k_value])
                    e -= (h_i[j*(Q-1) + j_value])
            energies.append(e)

        min_energy = min(energies)
        idx = energies.index(min_energy)
        if min_energy < cutoffs_list[idx]:
            predicted[i] = np.unique(train_labels)[idx]
            print(idx, predicted[i])
        else:
            predicted[i] = 100
    return predicted
