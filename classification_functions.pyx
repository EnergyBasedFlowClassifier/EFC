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
