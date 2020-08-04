import numpy as np

def local_fields(coupling_matrix, sitefreq, q):
    N = sitefreq.shape[0]
    fields = np.empty((N*(q-1)),dtype=float)

    for i in range(N):
        for ai in range(q-1):
            fields[i*(q-1) + ai] = sitefreq[i,ai]/sitefreq[i,q-1]
            for j in range(N):
                for aj in range(q-1):
                    fields[i*(q-1) + ai] /= coupling_matrix[i*(q-1) + ai, j*(q-1) + aj]**sitefreq[j,aj]
    return fields

def Sitefreq(encoded_msa, q, LAMBDA):
    nA = encoded_msa.shape[1]
    Meff = encoded_msa.shape[0]
    sitefreq = np.empty((nA,q),dtype=float)
    for i in range(nA):
        for aa in range(q):
            sitefreq[i,aa] = np.sum(np.equal(encoded_msa[:,i],aa))/Meff

    sitefreq = (1-LAMBDA)*sitefreq + LAMBDA/q
    return sitefreq

def Entropy(sitefreq,nA,nB,alpha):
    ent = np.zeros((nA+nB),dtype=float)
    for i in range(nA+nB):
        ent[i] = -np.sum((sitefreq[i,:]**alpha)*((sitefreq[i,:]**(1-alpha)-1)/(1-alpha)))

    return ent

def cantor(x, y):
    return (x + y) * (x + y + 1) / 2 + y

def Pairfreq(encoded_msa, sitefreq, q, LAMBDA):
    nP = encoded_msa.shape[1]
    Meff = encoded_msa.shape[0]
    pairfreq = np.zeros((nP,q,nP,q),dtype=float)
    print(pairfreq.shape)
    print(np.max(encoded_msa))
    for i in range(nP):
        for j in range(nP):
            c = cantor(encoded_msa[:,i],encoded_msa[:,j])
            unique,aaIdx = np.unique(c,True)
            for x,item in enumerate(unique):
                pairfreq[i, encoded_msa[aaIdx[x],i],j,encoded_msa[aaIdx[x],j]] = np.sum(np.equal(c,item))

    pairfreq /= Meff
    pairfreq = (1-LAMBDA)*pairfreq + LAMBDA/(q*q)

    for i in range(nP):
        for am_i in range(q):
            for am_j in range(q):
                if (am_i==am_j):
                    pairfreq[i,am_i,i,am_j] = sitefreq[i,am_i]
                else:
                    pairfreq[i,am_i,i,am_j] = 0.0
    return pairfreq

def Coupling(sitefreq, pairfreq, q):
    nP = sitefreq.shape[0]
    corr_matrix = np.empty(((nP)*(q-1), (nP)*(q-1)),dtype=float)
    for i in range(nP):
        for j in range(nP):
            for am_i in range(q-1):
                for am_j in range(q-1):
                    corr_matrix[i*(q-1)+am_i, j*(q-1)+am_j] = pairfreq[i,am_i,j,am_j] - sitefreq[i,am_i]*sitefreq[j,am_j]

    inv_corr = np.linalg.inv(corr_matrix)
    coupling_matrix = np.exp(np.negative(inv_corr))
    return coupling_matrix

def direct_information(coupling_matrix, sitefreq, q):
    nP = int(coupling_matrix.shape[0]/(q-1))
    tiny = 1.0e-100
    DI = np.zeros((nP,nP),dtype=float)

    fields = np.zeros(nP*(q-1))
    ent = np.zeros(nP,dtype=float)
    for i in range(nP):
        ent[i] -= np.sum(sitefreq[i,:]*np.log(sitefreq[i,:]))

    for i in range(nP-1):
        for j in range(i+1,nP):
            eij, h_i, h_j, Pi, Pj = local_fields(i, j, coupling_matrix, sitefreq, q)
            x = np.multiply(eij, np.outer(h_i, h_j))
            Pij = x/sum(sum(x))
            Pfac = np.outer(Pi, Pj)
            z = np.outer(Pij, np.log((Pij+tiny)/(Pfac+tiny)))
            DI[i,j] = np.trace(z)
            DI[j,i] = DI[i,j]
            fields[i*(q-1):(i+1)*(q-1)] += h_i[:(q-1)]

    average_DI = np.average(DI)
    for i in range(nP-1):
        average_i = np.average(DI[i,:])
        for j in range(i+1,nP):
            correction = (np.average(DI[:,j])*average_i)/average_DI
            DI[i,j] = DI[i,j] - correction
            DI[j,i] = DI[i,j]

    for i in range(nP):
        if ent[i] > (np.average(ent)+np.std(ent)):
            DI[i,:]=0
            DI[:,i]=0

    return DI, fields