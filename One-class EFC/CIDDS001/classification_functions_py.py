import numpy as np
from dca_functions_py import *

def create_oneclass_model(data,n_bins, lamb):
    sitefreq = Sitefreq(data, n_bins, lamb)
    pairfreq = Pairfreq(data, sitefreq, n_bins, lamb)
    couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
    h_i = local_fields(couplingmatrix, sitefreq, n_bins)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cutoff = define_cutoff(data, h_i, couplingmatrix, Q)
    return couplingmatrix, h_i, cutoff

def test_oneclass_model(test_data, model, h_i, expected, cutoff, n_bins):
    n_rows = test_data.shape[0]
    n_col = test_data.shape[1]

    energies = np.empty(n_rows, dtype=np.float64)
    predicted = np.empty(n_rows, dtype=bool)

    for i in range(n_rows):
        e = 0
        for j in range(n_col-1):
            j_value = test_data[i,j]
            if j_value != (n_bins-1):
                for k in range(j,n_col):
                    k_value = test_data[i,k]
                    if k_value != (n_bins-1):
                        e -= (model[j*(n_bins-1) + j_value, k*(n_bins-1) + k_value])
                e -= (h_i[j*(n_bins-1) + j_value])
        predicted[i] = e > cutoff
        energies[i] = e
    return predicted, energies


def define_cutoff(train_data, h_i, couplingmatrix, n_bins):
    n_rows = train_data.shape[0]
    n_col = train_data.shape[1]

    energies = np.empty(n_rows, dtype=np.float64)
    predicted = np.empty(n_rows, dtype=bool)

    for i in range(n_rows):
        e = 0
        for j in range(n_col-1):
            j_value = train_data[i,j]
            if j_value != (n_bins-1):
                for k in range(j,n_col):
                    k_value = train_data[i,k]
                    if k_value != (n_bins-1):
                        e -= (couplingmatrix[j*(n_bins-1) + j_value, k*(n_bins-1) + k_value])
                e -= (h_i[j*(n_bins-1) + j_value])
        energies[i] = e

    energies = np.sort(energies, axis=None)
    cutoff = energies[int(energies.shape[0]*0.95)]
    return cutoff
