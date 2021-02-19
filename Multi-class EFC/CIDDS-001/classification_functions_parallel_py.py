import numpy as np
from dca_functions import *
import numpy as np
from dca_functions import *
import time
import concurrent.futures
import multiprocessing
import itertools
import cProfile
import pstats

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

def create_multiclass_model(set, labels, n_bins, lamb):
    sitefreq = Sitefreq(set, n_bins, lamb)
    pairfreq = Pairfreq(set, sitefreq, n_bins, lamb)
    couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
    h_i = local_fields(couplingmatrix, sitefreq, n_bins)
    cutoff = define_cutoff(set, h_i, couplingmatrix, n_bins)

    return h_i, couplingmatrix, cutoff


def create_multiclass_model_parallel(data,labels,n_bins, lamb):
    start = time.time()
    data_per_type = []
    for label in np.unique(labels):
        selected = [data[i,:] for i in range(data.shape[0]) if labels[i] == label]
        data_per_type.append(np.array(selected))
    print("Tempo de separação das classes no treinamento: {}".format(time.time()-start))
    n_proc = len(data_per_type)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(create_multiclass_model, data_per_type, n_proc*[labels], n_proc*[n_bins], n_proc*[lamb])

    h_i_matrices = []
    coupling_matrices = []
    cutoffs_list = []
    for result in results:
        h_i_matrices.append(result[0])
        coupling_matrices.append(result[1])
        cutoffs_list.append(result[2])
    print(len(h_i_matrices))
    print("Tempo de treinamento: {}".format(time.time()-start))
    return h_i_matrices, coupling_matrices, cutoffs_list

def test_multiclass_model(test_data, h_i_matrices, coupling_matrices, cutoffs_list, n_bins, train_labels):
    n_rows = test_data.shape[0]
    n_columns = test_data.shape[1]
    n_classes = len(h_i_matrices)
    predicted = np.empty(test_data.shape[0],dtype=int)
    for i in range(n_rows):
        energies = np.empty(n_classes,dtype=float)
        for indx in range(n_classes):
            e = 0
            couplingmatrix = coupling_matrices[indx]
            h_i = h_i_matrices[indx]
            for j in range(n_columns-1):
                j_value = test_data[i,j]
                for k in range(j,n_columns):
                    k_value = test_data[i,k]
                    if j_value != (n_bins-1) and k_value != (n_bins-1):
                        e -= np.log(couplingmatrix[int(j*(n_bins-1) + j_value), int(k*(n_bins-1) + k_value)])
                if j_value != (n_bins-1):
                    e -= np.log(h_i[int(j*(n_bins-1) + j_value)])
            energies[indx] = e
        min_energy = np.min(energies)
        min_indx = np.where(energies==min_energy)[0]
        if min_energy < cutoffs_list[min_indx]:
            predicted[i] = train_labels[min_indx]
        else:
            predicted[i] = 100
    return predicted

def test_multiclass_model_parallel(test_data, h_i_matrices, coupling_matrices, cutoffs_list, n_bins, train_labels):
    start = time.time()
    n_jobs = multiprocessing.cpu_count()
    chunk_size = test_data.shape[0]//n_jobs
    print(chunk_size)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        data = [test_data[i*chunk_size:(i+1)*chunk_size] for i in range(n_jobs-1)]
        data.append(test_data[(n_jobs-1)*chunk_size::])
        results = executor.map(test_multiclass_model, data, n_jobs*[h_i_matrices], n_jobs*[coupling_matrices], n_jobs*[cutoffs_list], n_jobs*[n_bins], n_jobs*[train_labels])

    predicted = np.empty(0,dtype=int)
    for result in results:
        predicted = np.append(predicted, result, axis=0)
    print("Tempo de teste: {}".format(time.time()-start))
    return predicted
