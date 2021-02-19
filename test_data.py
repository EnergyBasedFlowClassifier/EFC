import numpy as np
from classification_functions import *

X_train = train_data
X_train_normals = train_data_normals
y_train = train_labels
X_test = test_data
y_test = test_labels

Q = 14 #max value in database
LAMBDA = 0.5

coupling, h_i, cutoff = create_oneclass_model(X_train_normals, Q, LAMBDA)
y_predicted, energies = test_oneclass_model(X_test, coupling, h_i, y_test, cutoff, Q)
precision = precision_score(y_test, y_predicted)
recall = recall_score(y_test, y_predicted)
