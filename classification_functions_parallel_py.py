import numpy as np
from dca_functions import *
import numpy as np
from dca_functions_py import *
import time
import concurrent.futures
import multiprocessing
import itertools
import cProfile
import pstats

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

def OneClassFit(train_data, Q, LAMBDA):
    sitefreq = Sitefreq(train_data, Q, LAMBDA)
    pairfreq = Pairfreq(train_data, sitefreq, Q, LAMBDA)
    couplingmatrix = Coupling(sitefreq, pairfreq, Q)
    h_i = LocalFields(couplingmatrix, sitefreq, Q)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cutoff = DefineCutoff(train_data, h_i, couplingmatrix, Q)
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


def FitClass(set, Q, LAMBDA):
    sitefreq = Sitefreq(set, Q, LAMBDA)
    pairfreq = Pairfreq(set, sitefreq, Q, LAMBDA)
    couplingmatrix = Coupling(sitefreq, pairfreq, Q)
    h_i = LocalFields(couplingmatrix, sitefreq, Q)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cutoff = DefineCutoff(set, h_i, couplingmatrix, Q)
    return h_i, couplingmatrix, cutoff

def MultiClassFit(data, labels, Q, LAMBDA):
    data_per_type = []
    for label in np.unique(labels):
        selected = [data[i,:] for i in range(data.shape[0]) if labels[i] == label]
        data_per_type.append(np.array(selected))
    n_jobs = len(data_per_type)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(FitClass, data_per_type, n_jobs*[Q], n_jobs*[LAMBDA])
    h_i_matrices = []
    coupling_matrices = []
    cutoffs_list = []
    for result in results:
        h_i_matrices.append(result[0])
        coupling_matrices.append(result[1])
        cutoffs_list.append(result[2])
    return h_i_matrices, coupling_matrices, cutoffs_list

def PredictSubset(test_data, h_i_matrices, coupling_matrices, cutoffs_list, Q, train_labels):
    n_inst = test_data.shape[0]
    n_attr = test_data.shape[1]
    n_classes = len(h_i_matrices)
    predicted = np.empty(test_data.shape[0],dtype=int)
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

def MultiClassPredict(test_data, h_i_matrices, coupling_matrices, cutoffs_list, Q, train_labels):
    n_jobs = multiprocessing.cpu_count()
    chunk_size = test_data.shape[0]//n_jobs
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = [test_data[i*chunk_size:(i+1)*chunk_size] for i in range(n_jobs-1)]
        data.append(test_data[(n_jobs-1)*chunk_size::])
        results = executor.map(PredictSubset, data, n_jobs*[h_i_matrices], n_jobs*[coupling_matrices], n_jobs*[cutoffs_list], n_jobs*[Q], n_jobs*[train_labels])

    predicted = np.empty(0, dtype=int)
    for result in results:
        predicted = np.append(predicted, result, axis=0)
    return predicted
