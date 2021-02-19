# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1

import numpy as np
from dca_functions import *
cimport numpy as np
cimport cython

import line_profiler
import distutils
import time

DTYPE = np.int
ctypedef np.int_t DTYPE_t


def define_cutoff(np.ndarray[DTYPE_t, ndim=2] train_data, np.ndarray[double, ndim=1] h_i, np.ndarray[double, ndim=2] couplingmatrix, int n_bins):
    cdef np.ndarray[double, ndim=1] energies = np.empty(train_data.shape[0], dtype=float)
    cdef int i, j, k
    cdef float e
    for i in range(train_data.shape[0]):
        e = 0
        for j in range(train_data.shape[1]-1):
            j_value = train_data[i,j]
            for k in range(j,train_data.shape[1]):
                k_value = train_data[i,k]
                if j_value != (n_bins-1) and k_value != (n_bins-1):
                    e -= np.log(couplingmatrix[int(j*(n_bins-1) + j_value), int(k*(n_bins-1) + k_value)])
            if j_value != (n_bins-1):
                e -= np.log(h_i[int(j*(n_bins-1) + j_value)])
        energies[i] = e

    energies = np.sort(energies, axis=None)
    cutoff = energies[int(energies.shape[0]*0.95)]
    return cutoff

def create_oneclass_model(np.ndarray[DTYPE_t, ndim=2] data, int n_bins, float lamb):
    cdef np.ndarray[double, ndim=2] sitefreq = Sitefreq(data, n_bins, lamb)
    cdef np.ndarray[double, ndim=4] pairfreq = Pairfreq(data, sitefreq, n_bins, lamb)
    cdef np.ndarray[double, ndim=2] couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
    cdef np.ndarray[double, ndim=1] h_i = local_fields(couplingmatrix, sitefreq, n_bins)

    return couplingmatrix, h_i

def test_oneclass_model(np.ndarray[DTYPE_t, ndim=2] test_data, np.ndarray[double, ndim=2] model, np.ndarray[double, ndim=1] h_i, np.ndarray[DTYPE_t, ndim=1] expected, float cutoff, int n_bins):
    cdef np.ndarray[double, ndim=1] energies = np.empty(test_data.shape[0], dtype=float)
    cdef np.ndarray[DTYPE_t, ndim=1] predicted = np.empty(test_data.shape[0], dtype=int)
    cdef int i, j, k
    cdef float e

    for i in range(test_data.shape[0]):
        e = 0
        for j in range(test_data.shape[1]-1):
            j_value = test_data[i,j]
            for k in range(j,test_data.shape[1]):
                k_value = test_data[i,k]
                if j_value != (n_bins-1) and k_value != (n_bins-1):
                    e -= np.log(model[int(j*(n_bins-1) + j_value), int(k*(n_bins-1) + k_value)])
            if j_value != (n_bins-1):
                e -= np.log(h_i[int(j*(n_bins-1) + j_value)])
        if e <= cutoff:
            predicted[i] = 0
        if e > cutoff:
            predicted[i] = 1
        energies[i] = e
    return predicted, energies


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def create_multiclass_model_seq(data,labels,n_bins, lamb):
    cdef int n_classes = np.unique(labels).shape[0]
    cdef np.ndarray[object, ndim=1] data_per_type = np.empty(n_classes, dtype=np.ndarray)
    cdef int label
    for label in range(n_classes):
      data_per_type[label] = np.array([data[i,:] for i in range(data.shape[0]) if labels[i] == label])

    cdef np.ndarray[double, ndim=2] sitefreq
    cdef np.ndarray[double, ndim=4] pairfreq
    cdef np.ndarray[double, ndim=2] couplingmatrix
    cdef np.ndarray[double, ndim=1] h_i
    cdef np.ndarray[object, ndim=1] h_i_matrices = np.empty(n_classes, dtype=np.ndarray)
    cdef np.ndarray[object, ndim=1] coupling_matrices = np.empty(n_classes, dtype=np.ndarray)
    cdef np.ndarray[double, ndim=1] cutoffs_list = np.empty(n_classes, dtype=float)

    for label in range(n_classes):
      sitefreq = Sitefreq(data_per_type[label], n_bins, lamb)
      pairfreq = Pairfreq(data_per_type[label], sitefreq, n_bins, lamb)
      couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
      h_i = local_fields(couplingmatrix, sitefreq, n_bins)

      coupling_matrices[label] = couplingmatrix
      h_i_matrices[label] = h_i
      cutoffs_list[label] = define_cutoff(data_per_type[label], h_i, couplingmatrix, n_bins)

    return h_i_matrices, coupling_matrices, cutoffs_list

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
def test_multiclass_model_seq(np.ndarray[DTYPE_t, ndim=2] test_data, np.ndarray[object, ndim=1] h_i_matrices, np.ndarray[object, ndim=1] coupling_matrices, np.ndarray[double, ndim=1] cutoffs_list, int n_bins, np.ndarray[DTYPE_t, ndim=1] train_labels):
    cdef int n_rows = test_data.shape[0]
    cdef int n_columns = test_data.shape[1]
    cdef int n_classes = h_i_matrices.shape[0]
    cdef np.ndarray result = np.empty(n_rows, dtype=int)
    cdef int i, label, j, k, j_value, k_value, predicted_label
    cdef double e
    cdef np.ndarray[double, ndim=2] couplingmatrix
    cdef np.ndarray[double, ndim=1] h_i
    for i in range(n_rows):
        min_energy = None
        for label in range(n_classes):
          couplingmatrix = coupling_matrices[label]
          h_i = h_i_matrices[label]
          e = 0
          for j in range(n_columns-1):
              j_value = int(test_data[i, j])
              for k in range(j,n_columns):
                  k_value = int(test_data[i, k])
                  if j_value != (n_bins-1) and k_value != (n_bins-1):
                      e -= np.log(couplingmatrix[j*(n_bins-1) + j_value, k*(n_bins-1)+k_value])
              if j_value != (n_bins-1):
                  e -= np.log(h_i[j*(n_bins-1) + j_value])
          if min_energy is None or e < min_energy:
              min_energy = e
              predicted_label = label

        if min_energy < cutoffs_list[predicted_label]:
            result[i] = predicted_label
        else:
            result[i] = 100
    return result
