import numpy as np
cimport numpy as np
cimport cython
from scipy import spatial

DTYPE = 'int'
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def Weights(np.ndarray[DTYPE_t, ndim=2] data, float THETA):
    hammdist = spatial.distance.pdist(data, 'hamming')
    weight_matrix = spatial.distance.squareform(hammdist < (1.0- THETA))
    weight = 1.0 / (np.sum(weight_matrix, axis = 1) + 1.0)
    return weight


@cython.boundscheck(False)
@cython.wraparound(False)
def Sitefreq(np.ndarray[DTYPE_t, ndim=2] data, int Q, float LAMBDA):
    cdef int n_attr = data.shape[1]
    cdef np.ndarray[double, ndim=2] sitefreq = np.empty((n_attr, Q),dtype='float')
    cdef int i, aa
    for i in range(n_attr):
        for aa in range(Q):
            sitefreq[i,aa] = np.sum(np.equal(data[:,i],aa))

    sitefreq /= data.shape[0]
    sitefreq = (1-LAMBDA)*sitefreq + LAMBDA/Q
    return sitefreq

@cython.boundscheck(False)
@cython.wraparound(False)
def cantor(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    return (x + y) * (x + y + 1) / 2 + y

@cython.boundscheck(False)
@cython.wraparound(False)
def Pairfreq(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[double, ndim=2] sitefreq, int Q, float LAMBDA):
    cdef int n_attr = data.shape[1]
    cdef DTYPE_t[:,:] data_view = data
    cdef np.ndarray[double, ndim=4] pairfreq = np.zeros((n_attr, Q, n_attr, Q),dtype='float')
    cdef double[:,:,:,:] pairfreqview = pairfreq

    cdef int i, j, x, am_i, am_j
    cdef float item
    cdef np.ndarray[double, ndim=1] unique, c
    cdef np.ndarray[DTYPE_t, ndim=1] aaIdx

    for i in range(n_attr):
        for j in range(n_attr):
            c = cantor(data[:,i],data[:,j])
            unique,aaIdx = np.unique(c,True)
            for x,item in enumerate(unique):
                pairfreqview[i, data_view[aaIdx[x],i],j,data_view[aaIdx[x],j]] = np.sum(np.equal(c,item))

    pairfreq /= data.shape[0]
    pairfreq = (1-LAMBDA)*pairfreq + LAMBDA/(Q*Q)

    for i in range(n_attr):
        for am_i in range(Q):
            for am_j in range(Q):
                if (am_i==am_j):
                    pairfreq[i,am_i,i,am_j] = sitefreq[i,am_i]
                else:
                    pairfreq[i,am_i,i,am_j] = 0.0
    return pairfreq

@cython.boundscheck(False)
@cython.wraparound(False)
def LocalFields(np.ndarray[double, ndim=2] coupling_matrix, np.ndarray[double, ndim=2] sitefreq, int Q):
    cdef int n_inst = sitefreq.shape[0]
    cdef np.ndarray[double, ndim=1] fields = np.empty((n_inst*(Q-1)),dtype='float')
    cdef int i, ai, j, aj

    for i in range(n_inst):
        for ai in range(Q-1):
            fields[i*(Q-1) + ai] = sitefreq[i,ai]/sitefreq[i,Q-1]
            for j in range(n_inst):
                for aj in range(Q-1):
                    fields[i*(Q-1) + ai] /= coupling_matrix[i*(Q-1) + ai, j*(Q-1) + aj]**sitefreq[j,aj]
    return fields

@cython.boundscheck(False)
@cython.wraparound(False)
def Coupling(np.ndarray[double, ndim=2] sitefreq, np.ndarray[double, ndim=4] pairfreq, int Q):
    cdef int n_attr = sitefreq.shape[0]
    cdef np.ndarray[double, ndim=2] corr_matrix = np.empty(((n_attr)*(Q-1), (n_attr)*(Q-1)),dtype='float')
    cdef int i, j, am_i, am_j
    for i in range(n_attr):
        for j in range(n_attr):
            for am_i in range(Q-1):
                for am_j in range(Q-1):
                    corr_matrix[i*(Q-1)+am_i, j*(Q-1)+am_j] = pairfreq[i,am_i,j,am_j] - sitefreq[i,am_i]*sitefreq[j,am_j]

    cdef np.ndarray[double, ndim=2] inv_corr = np.linalg.inv(corr_matrix)
    cdef np.ndarray[double, ndim=2] coupling_matrix = np.exp(np.negative(inv_corr))
    return coupling_matrix
