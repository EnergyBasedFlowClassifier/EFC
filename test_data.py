import numpy as np
from classification_functions_parallel import *
from classification_functions_seq import *

X_train = train_data
X_train_normals = train_data_normals
y_train = train_labels
X_test = test_data
y_test = test_labels

Q = 14 #max value in database
LAMBDA = 0.5

#One class EFC
coupling, h_i, cutoff = create_oneclass_model(X_train_normals, Q, LAMBDA)
y_predicted, energies = test_oneclass_model(X_test, coupling, h_i, y_test, cutoff, Q)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)

#Multi class EFC
coupling_matrices, h_i_matrices, cutoffs_list = create_multiclass_model_seq(X_train, y_train, Q, LAMBDA)
y_predicted = test_multiclass_model_seq(X_test, h_i_matrices, coupling_matrices, cutoffs_list, Q, y_train)

#Multi class EFC using parallelism
coupling_matrices, h_i_matrices, cutoffs_list = create_multiclass_model_parallel(X_train, y_train, Q, LAMBDA)
y_predicted = test_multiclass_model_parallel(X_test, h_i_matrices, coupling_matrices, cutoffs_list, Q, y_train)

precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
