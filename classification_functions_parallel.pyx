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


def create_multiclass_model(np.ndarray[DTYPE_t, ndim=2] set, int n_bins, float lamb):
    cdef np.ndarray[double, ndim=2] sitefreq = Sitefreq(set, n_bins, lamb)
    cdef np.ndarray[double, ndim=4] pairfreq = Pairfreq(set, sitefreq, n_bins, lamb)
    cdef np.ndarray[double, ndim=2] couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
    cdef np.ndarray[double, ndim=1] h_i = local_fields(couplingmatrix, sitefreq, n_bins)
    cdef double cutoff = define_cutoff(set, h_i, couplingmatrix, n_bins)
    return h_i, couplingmatrix, cutoff


def create_multiclass_model_parallel(np.ndarray[DTYPE_t, ndim=2] data, np.ndarray[DTYPE_t, ndim=1] labels, int n_bins, float lamb):
    cdef int n_classes = np.unique(labels).shape[0]
    cdef np.ndarray[object, ndim=1] data_per_type = np.empty(n_classes, dtype=np.ndarray)
    cdef int label
    for indx, label in enumerate(np.unique(labels)):
      data_per_type[indx] = np.array([data[i,:] for i in range(data.shape[0]) if labels[i] == label])

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(create_multiclass_model, data_per_type, n_classes*[n_bins], n_classes*[lamb])

    cdef np.ndarray[object, ndim=1] h_i_matrices = np.empty(n_classes, dtype=np.ndarray)
    cdef np.ndarray[object, ndim=1] coupling_matrices = np.empty(n_classes, dtype=np.ndarray)
    cdef np.ndarray[double, ndim=1] cutoffs_list = np.empty(n_classes, dtype=float)

    for indx, result in enumerate(results):
        h_i_matrices[indx] = result[0]
        coupling_matrices[indx] = result[1]
        cutoffs_list[indx] = result[2]

    return h_i_matrices, coupling_matrices, cutoffs_list


def test_multiclass_model(np.ndarray[DTYPE_t, ndim=2] test_data, np.ndarray[object, ndim=1] h_i_matrices, np.ndarray[object, ndim=1] coupling_matrices, np.ndarray[double, ndim=1] cutoffs_list, int n_bins, np.ndarray[DTYPE_t, ndim=1] train_labels):
    cdef int n_rows = test_data.shape[0]
    cdef int n_columns = test_data.shape[1]
    cdef int n_classes = h_i_matrices.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=1] result = np.empty(n_rows, dtype=int)
    cdef int i, label, j, k, j_value, k_value, predicted_label
    cdef double e
    cdef np.ndarray[double, ndim=2] couplingmatrix
    cdef np.ndarray[double, ndim=1] h_i
    for i in range(n_rows):
        min_energy = None
        for indx, label in enumerate(train_labels):
          couplingmatrix = coupling_matrices[indx]
          h_i = h_i_matrices[indx]
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
              predicted_index = indx

        if min_energy < cutoffs_list[predicted_index]:
            result[i] = predicted_label
        else:
            result[i] = 100
    return result


def test_multiclass_model_parallel(np.ndarray[DTYPE_t, ndim=2] test_data, np.ndarray[object, ndim=1] h_i_matrices, np.ndarray[object, ndim=1] coupling_matrices, np.ndarray[double, ndim=1] cutoffs_list, int n_bins, np.ndarray[DTYPE_t, ndim=1] train_labels):
    cdef double start = time.time()
    cdef int n_jobs = multiprocessing.cpu_count()
    cdef int chunk_size = test_data.shape[0]//n_jobs
    cdef int i
    cdef np.ndarray[object, ndim=1] data_frac = np.empty(n_jobs, dtype=np.ndarray)
    for i in range(n_jobs-1):
      data_frac[i] = test_data[i*chunk_size:(i+1)*chunk_size]
    data_frac[i+1] = test_data[(n_jobs-1)*chunk_size::]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(test_multiclass_model, data_frac, n_jobs*[h_i_matrices], n_jobs*[coupling_matrices], n_jobs*[cutoffs_list], n_jobs*[n_bins], n_jobs*[train_labels])

    predicted = []

    for result in results:
        predicted += list(result)
    print("Tempo de teste: {}".format(time.time()-start))
    return predicted
