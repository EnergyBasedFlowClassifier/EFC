import numpy as np
cimport numpy as np
cimport cython
from scipy import spatial

DTYPE = 'int'
ctypedef np.int_t DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
def Weights(np.ndarray[DTYPE_t, ndim=2] encoded_msa, float THETA):
    hammdist = spatial.distance.pdist(encoded_msa, 'hamming')
    weight_matrix = spatial.distance.squareform(hammdist < (1.0- THETA))
    weight = 1.0 / (np.sum(weight_matrix, axis = 1) + 1.0)
    return weight


@cython.boundscheck(False)
@cython.wraparound(False)
def Sitefreq(np.ndarray[DTYPE_t, ndim=2] encoded_msa, np.ndarray[double, ndim=1] weights, int q, float LAMBDA):
    cdef int nA = encoded_msa.shape[1]
    cdef np.ndarray[double, ndim=2] sitefreq = np.empty((nA,q),dtype='float')
    cdef int i, aa
    for i in range(nA):
        for aa in range(q):
            sitefreq[i,aa] = np.sum(np.equal(encoded_msa[:,i],aa)*weights)

    sitefreq /= np.sum(weights)
    sitefreq = (1-LAMBDA)*sitefreq + LAMBDA/q
    return sitefreq

@cython.boundscheck(False)
@cython.wraparound(False)
def cantor(np.ndarray[DTYPE_t, ndim=1] x, np.ndarray[DTYPE_t, ndim=1] y):
    return (x + y) * (x + y + 1) / 2 + y

@cython.boundscheck(False)
@cython.wraparound(False)
def Pairfreq(np.ndarray[DTYPE_t, ndim=2] encoded_msa, np.ndarray[double, ndim=2] sitefreq, np.ndarray[double, ndim=1] weights, int q, float LAMBDA):
    cdef int nP = encoded_msa.shape[1]

    cdef DTYPE_t[:,:] encoded_msa_view = encoded_msa
    cdef np.ndarray[double, ndim=4] pairfreq = np.zeros((nP,q,nP,q),dtype='float')
    cdef double[:,:,:,:] pairfreqview = pairfreq

    cdef int i, j, x, am_i, am_j
    cdef float item
    cdef np.ndarray[double, ndim=1] unique, c
    cdef np.ndarray[DTYPE_t, ndim=1] aaIdx

    for i in range(nP):
        for j in range(nP):
            c = cantor(encoded_msa[:,i],encoded_msa[:,j])
            unique,aaIdx = np.unique(c,True)
            for x,item in enumerate(unique):
                pairfreqview[i, encoded_msa_view[aaIdx[x],i],j,encoded_msa_view[aaIdx[x],j]] = np.sum(np.equal(c,item)*weights)

    pairfreq /= np.sum(weights)
    pairfreq = (1-LAMBDA)*pairfreq + LAMBDA/(q*q)

    for i in range(nP):
        for am_i in range(q):
            for am_j in range(q):
                if (am_i==am_j):
                    pairfreq[i,am_i,i,am_j] = sitefreq[i,am_i]
                else:
                    pairfreq[i,am_i,i,am_j] = 0.0
    return pairfreq

@cython.boundscheck(False)
@cython.wraparound(False)
def LocalFields(np.ndarray[double, ndim=2] coupling_matrix, np.ndarray[double, ndim=2] sitefreq, int q):
    cdef int N = sitefreq.shape[0]
    cdef np.ndarray[double, ndim=1] fields = np.empty((N*(q-1)),dtype='float')
    cdef int i, ai, j, aj

    for i in range(N):
        for ai in range(q-1):
            fields[i*(q-1) + ai] = sitefreq[i,ai]/sitefreq[i,q-1]
            for j in range(N):
                for aj in range(q-1):
                    fields[i*(q-1) + ai] /= coupling_matrix[i*(q-1) + ai, j*(q-1) + aj]**sitefreq[j,aj]
    return fields

@cython.boundscheck(False)
@cython.wraparound(False)
def Coupling(np.ndarray[double, ndim=2] sitefreq, np.ndarray[double, ndim=4] pairfreq, int q):
    cdef int nP = sitefreq.shape[0]
    cdef np.ndarray[double, ndim=2] corr_matrix = np.empty(((nP)*(q-1), (nP)*(q-1)),dtype='float')
    cdef int i, j, am_i, am_j
    for i in range(nP):
        for j in range(nP):
            for am_i in range(q-1):
                for am_j in range(q-1):
                    corr_matrix[i*(q-1)+am_i, j*(q-1)+am_j] = pairfreq[i,am_i,j,am_j] - sitefreq[i,am_i]*sitefreq[j,am_j]

    cdef np.ndarray[double, ndim=2] inv_corr = np.linalg.inv(corr_matrix)
    cdef np.ndarray[double, ndim=2] coupling_matrix = np.exp(np.negative(inv_corr))
    return coupling_matrix
