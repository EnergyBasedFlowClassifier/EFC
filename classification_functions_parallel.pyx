import numpy as np
from dca_functions import *
import time
import concurrent.futures
import multiprocessing
import itertools
cimport numpy as np
cimport cython
import time

DTYPE = np.int
ctypedef np.int_t DTYPE_t
def DefineCutoff(DTYPE_t[:,:] train_data, double[:] h_i, double[:,:] couplingmatrix, int Q):
    cdef int n_inst = train_data.shape[0]
    cdef int n_attr = train_data.shape[1]
    cdef double[:] energies = np.empty(n_inst, dtype= 'float64')
    cdef int i, j, k, k_value, j_value
    cdef float e
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

def OneClassFit(np.ndarray[DTYPE_t, ndim=2] data, int Q, float LAMBDA, float THETA):
    cdef np.ndarray[double, ndim=1] weights = Weights(data, THETA)
    cdef np.ndarray[double, ndim=2] sitefreq = Sitefreq(data, weights, Q, LAMBDA)
    cdef np.ndarray[double, ndim=4] pairfreq = Pairfreq(data, sitefreq, weights, Q, LAMBDA)
    cdef np.ndarray[double, ndim=2] couplingmatrix = Coupling(sitefreq, pairfreq, Q)
    cdef np.ndarray[double, ndim=1] h_i = LocalFields(couplingmatrix, sitefreq, Q)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cdef double cutoff = DefineCutoff(data, h_i, couplingmatrix, Q)
    return couplingmatrix, h_i, cutoff


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def OneClassPredict(DTYPE_t[:,:] test_data, double[:,:] couplingmatrix, double[:] h_i, float cutoff, int Q):
    cdef int n_inst = test_data.shape[0]
    cdef int n_attr = test_data.shape[1]
    cdef double[:] energies = np.empty(n_inst, dtype= 'float64')
    cdef DTYPE_t[:] predicted = np.empty(n_inst, dtype= 'int')
    cdef int i, j, k, k_value, j_value
    cdef float e
    for i in range(n_inst):
        e = 0
        for j in range(n_attr-1):
            j_value = test_data[i,j]
            if j_value != (Q-1):
                for k in range(j,n_attr):
                    k_value = test_data[i,k]
                    if k_value != (Q-1):
                        e -= (couplingmatrix[j*(Q-1) + j_value, k*(Q-1) + k_value])
                e -= (h_i[j*(Q-1) + j_value])
        predicted[i] = e > cutoff
        energies[i] = e
    return np.asarray(predicted), np.asarray(energies)


def FitClass(np.ndarray[DTYPE_t, ndim=2] set, int Q, float LAMBDA, float THETA):
    cdef np.ndarray[double, ndim=1] weights = Weights(set, THETA)
    cdef np.ndarray[double, ndim=2] sitefreq = Sitefreq(set, weights, Q, LAMBDA)
    cdef np.ndarray[double, ndim=4] pairfreq = Pairfreq(set, sitefreq, weights, Q, LAMBDA)
    cdef np.ndarray[double, ndim=2] couplingmatrix = Coupling(sitefreq, pairfreq, Q)
    cdef np.ndarray[double, ndim=1] h_i = LocalFields(couplingmatrix, sitefreq, Q)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cdef double cutoff = DefineCutoff(set, h_i, couplingmatrix, Q)
    return h_i, couplingmatrix, cutoff


def MultiClassFit(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=1] labels, int Q, float LAMBDA, float THETA):
    cdef int n_classes = np.unique(labels).shape[0]
    cdef np.ndarray[object, ndim=1] data_per_type = np.empty(n_classes, dtype=np.ndarray)
    cdef int label
    for indx, label in enumerate(np.unique(labels)):
      data_per_type[indx] = np.array([data[i,:] for i in range(data.shape[0]) if labels[i] == label])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(FitClass, data_per_type, n_classes*[Q], n_classes*[LAMBDA], n_classes*[THETA])

    cdef np.ndarray[object, ndim=1] h_i_matrices = np.empty(n_classes, dtype=np.ndarray)
    cdef np.ndarray[object, ndim=1] coupling_matrices = np.empty(n_classes, dtype=np.ndarray)
    cdef np.ndarray[double, ndim=1] cutoffs_list = np.empty(n_classes, dtype=float)
    for indx, result in enumerate(results):
        h_i_matrices[indx] = result[0]
        coupling_matrices[indx] = result[1]
        cutoffs_list[indx] = result[2]

    return h_i_matrices, coupling_matrices, cutoffs_list


def PredictSubset(np.ndarray[DTYPE_t, ndim=2] test_data, np.ndarray[object, ndim=1] h_i_matrices, np.ndarray[object, ndim=1] coupling_matrices, np.ndarray[double, ndim=1] cutoffs_list, int Q, np.ndarray[DTYPE_t, ndim=1] train_labels):
    cdef int n_inst = test_data.shape[0]
    cdef int n_attr = test_data.shape[1]
    cdef int n_classes = h_i_matrices.shape[0]
    cdef np.ndarray predicted = np.empty(n_inst, dtype=int)
    cdef int i, label, j, k, j_value, k_value, idx
    cdef double e, min_energy
    cdef np.ndarray[double, ndim=2] couplingmatrix
    cdef np.ndarray[double, ndim=1] h_i
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
        else:
            predicted[i] = 100
    return predicted


def MultiClassPredict(np.ndarray[DTYPE_t, ndim=2] test_data, np.ndarray[object, ndim=1] h_i_matrices, np.ndarray[object, ndim=1] coupling_matrices, np.ndarray[double, ndim=1] cutoffs_list, int Q, np.ndarray[DTYPE_t, ndim=1] train_labels):
    cdef int n_jobs = multiprocessing.cpu_count()
    cdef int chunk_size = test_data.shape[0]//n_jobs
    cdef int i
    cdef np.ndarray[object, ndim=1] data_frac = np.empty(n_jobs, dtype=np.ndarray)
    for i in range(n_jobs-1):
      data_frac[i] = test_data[i*chunk_size:(i+1)*chunk_size]
    data_frac[i+1] = test_data[(n_jobs-1)*chunk_size::]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(PredictSubset, data_frac, n_jobs*[h_i_matrices], n_jobs*[coupling_matrices], n_jobs*[cutoffs_list], n_jobs*[Q], n_jobs*[train_labels])

    predicted = []
    for result in results:
        predicted += list(result)

    return predicted
