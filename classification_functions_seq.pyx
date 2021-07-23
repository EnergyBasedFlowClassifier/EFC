import numpy as np
cimport numpy as np
from dca_functions import *
cimport cython

DTYPE =  'int'
ctypedef np.int_t DTYPE_t
ctypedef np.uint8_t uint8
# cython: profile=True
import line_profiler

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
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

def OneClassFit(np.ndarray[DTYPE_t, ndim=2] data, int Q, float LAMBDA):
    cdef np.ndarray[double, ndim=2] sitefreq = Sitefreq(data, Q, LAMBDA)
    cdef np.ndarray[double, ndim=4] pairfreq = Pairfreq(data, sitefreq, Q, LAMBDA)
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
    cdef DTYPE_t[:] predicted = np.empty(n_inst, dtype= 'int'
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


@cython.boundscheck(False)
@cython.wraparound(False)
def MultiClassFit(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=1] labels, DTYPE_t Q, float LAMBDA, float THETA):
    cdef np.ndarray[DTYPE_t, ndim=2] data_concat = np.empty((data.shape[0],data.shape[1]+1), dtype=DTYPE)
    data_concat[:,:-1] = data
    data_concat[:, -1] = labels

    cdef int n_classes = np.unique(labels).shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] subset
    cdef np.ndarray[double, ndim=2] sitefreq
    cdef np.ndarray[double, ndim=4] pairfreq
    cdef np.ndarray[double, ndim=2] couplingmatrix
    cdef np.ndarray[double, ndim=1] h_i
    cdef double cutoff

    cdef np.ndarray[object, ndim=1] h_i_matrices = np.empty(n_classes, dtype=np.ndarray)
    cdef np.ndarray[object, ndim=1] coupling_matrices = np.empty(n_classes, dtype=np.ndarray)
    cdef np.ndarray[double, ndim=1] cutoffs_list = np.empty(n_classes, dtype=float)

    for idx, label in enumerate(np.unique(labels)):
      subset = data_concat[data_concat[:,-1] == label]
      weights = Weights(subset, THETA)
      sitefreq = Sitefreq(subset, weights, Q, LAMBDA)
      pairfreq = Pairfreq(subset, sitefreq, weights , Q, LAMBDA)
      couplingmatrix = Coupling(sitefreq, pairfreq, Q)
      h_i = LocalFields(couplingmatrix, sitefreq, Q)
      couplingmatrix = np.log(couplingmatrix)
      h_i = np.log(h_i)
      cutoff = DefineCutoff(subset, h_i, couplingmatrix, Q)
      coupling_matrices[idx] = couplingmatrix
      h_i_matrices[idx] = h_i
      cutoffs_list[idx] = cutoff

    return h_i_matrices, coupling_matrices, cutoffs_list

@cython.boundscheck(False)
@cython.wraparound(False)
def MultiClassPredict(np.ndarray[DTYPE_t, ndim=2] test_data, np.ndarray[object, ndim=1] h_i_matrices, np.ndarray[object, ndim=1] coupling_matrices, np.ndarray[double, ndim=1] cutoffs_list, int Q, np.ndarray[DTYPE_t, ndim=1] train_labels):
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
