import numpy as np

def Weights(np.ndarray[DTYPE_t, ndim=2] data, float THETA):
    hammdist = spatial.distance.pdist(data, 'hamming')
    weight_matrix = spatial.distance.squareform(hammdist < (1.0- THETA))
    weight = 1.0 / (np.sum(weight_matrix, axis = 1) + 1.0)
    return weight

def Sitefreq(data, weights, Q, LAMBDA):
    n_attr = data.shape[1]
    sitefreq = np.empty((n_attr, Q),dtype=float)
    for i in range(n_attr):
        for aa in range(Q):
            sitefreq[i,aa] = np.sum(np.equal(data[:,i],aa)*weights)
            
    sitefreq /= np.sum(weights)
    sitefreq = (1-LAMBDA)*sitefreq + LAMBDA/Q
    return sitefreq

def Pairfreq(data, sitefreq, weights, Q, LAMBDA):
    n_attr = data.shape[1]
    pairfreq = np.zeros((n_attr, Q, n_attr, Q),dtype=float)

    for i in range(n_attr):
        for j in range(n_attr):
            c = cantor(data[:,i], data[:,j])
            unique, aaIdx = np.unique(c, True)
            for x, item in enumerate(unique):
                pairfreq[i, data[aaIdx[x],i],j,data[aaIdx[x],j]] = np.sum(np.equal(c,item)*weights)
    pairfreq /= np.sum(weights)
    pairfreq = (1-LAMBDA)*pairfreq + LAMBDA/(Q*Q)

    for i in range(n_attr):
        for am_i in range(Q):
            for am_j in range(Q):
                if (am_i==am_j):
                    pairfreq[i,am_i,i,am_j] = sitefreq[i,am_i]
                else:
                    pairfreq[i,am_i,i,am_j] = 0.0
    return pairfreq

def cantor(x, y):
    return (x + y) * (x + y + 1) / 2 + y

def LocalFields(coupling_matrix, sitefreq, Q):
    n_inst = sitefreq.shape[0]
    fields = np.empty((n_inst*(Q-1)),dtype=float)

    for i in range(n_inst):
        for ai in range(Q-1):
            fields[i*(Q-1) + ai] = sitefreq[i,ai]/sitefreq[i,Q-1]
            for j in range(n_inst):
                for aj in range(Q-1):
                    fields[i*(Q-1) + ai] /= coupling_matrix[i*(Q-1) + ai, j*(Q-1) + aj]**sitefreq[j,aj]
    return fields


def Coupling(sitefreq, pairfreq, q):
    n_inst = sitefreq.shape[0]
    corr_matrix = np.empty(((nP)*(Q-1), (nP)*(Q-1)),dtype=float)
    for i in range(nP):
        for j in range(nP):
            for am_i in range(Q-1):
                for am_j in range(Q-1):
                    corr_matrix[i*(Q-1)+am_i, j*(Q-1)+am_j] = pairfreq[i,am_i,j,am_j] - sitefreq[i,am_i]*sitefreq[j,am_j]

    inv_corr = np.linalg.inv(corr_matrix)
    coupling_matrix = np.exp(np.negative(inv_corr))
    return coupling_matrix
