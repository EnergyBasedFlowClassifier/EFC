import numpy as np
cimport numpy as np
from dca_functions import *
cimport cython
import math

DTYPE =  'int'
ctypedef  np.int_t DTYPE_t
ctypedef np.uint8_t uint8

def create_oneclass_model(np.ndarray[DTYPE_t, ndim=2] data, int n_bins, float lamb):
    cdef np.ndarray[double, ndim=2] sitefreq = Sitefreq(data, n_bins, lamb)
    cdef np.ndarray[double, ndim=4] pairfreq = Pairfreq(data, sitefreq, n_bins, lamb)
    cdef np.ndarray[double, ndim=2] couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
    cdef np.ndarray[double, ndim=1] h_i = local_fields(couplingmatrix, sitefreq, n_bins)
    couplingmatrix = np.log(couplingmatrix)
    h_i = np.log(h_i)
    cdef double cutoff = define_cutoff(data, h_i, couplingmatrix, n_bins)

    return couplingmatrix, h_i, cutoff

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def define_cutoff(DTYPE_t[:,:] train_data, double[:] h_i, double[:,:] couplingmatrix, int n_bins):
    cdef int n_rows = train_data.shape[0]
    cdef int n_col = train_data.shape[1]

    cdef double[:] energies = np.empty(n_rows, dtype= 'float64')

    cdef int i, j, k, k_value, j_value
    cdef float e


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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def test_oneclass_model(DTYPE_t[:,:] test_data, double[:,:] couplingmatrix, double[:] h_i, np.ndarray[DTYPE_t, ndim=1] expected, float cutoff, int n_bins):
    cdef int n_rows = test_data.shape[0]
    cdef int n_col = test_data.shape[1]

    cdef double[:] energies = np.empty(n_rows, dtype= 'float64')
    cdef DTYPE_t[:] predicted = np.empty(n_rows, dtype= 'int')

    cdef int i, j, k, k_value, j_value
    cdef float e

    for i in range(n_rows):
        e = 0
        for j in range(n_col-1):
            j_value = test_data[i,j]
            if j_value != (n_bins-1):
                for k in range(j,n_col):
                    k_value = test_data[i,k]
                    if k_value != (n_bins-1):
                        e -= (couplingmatrix[j*(n_bins-1) + j_value, k*(n_bins-1) + k_value])
                e -= (h_i[j*(n_bins-1) + j_value])

        predicted[i] = e > cutoff
        energies[i] = e
    return np.asarray(predicted), np.asarray(energies)
