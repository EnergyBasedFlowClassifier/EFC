from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, KBinsDiscretizer, Normalizer, MaxAbsScaler
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

#group continuos and symbolic features indexes
symbolic = [1, 6, 8]
continuous = [x for x in range(8) if x not in symbolic]

# load data
for fold in range(1,6):
    train = np.array(pd.read_csv("5-fold_sets/Raw/Sets{}/reduced_train.csv".format(fold), header=None))
    test = np.array(pd.read_csv("5-fold_sets/Raw/Sets{}/test.csv".format(fold), header=None))


    #encode symbolic features
    enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
    enc.fit(train[:, symbolic])
    train[:, symbolic] = enc.transform(train[:, symbolic])
    test[:, symbolic] = enc.transform(test[:, symbolic])
    test[:, symbolic] = np.nan_to_num(test[:, symbolic].astype('float'), nan=np.max(test[:, symbolic])+1)
    print(enc.categories_)

    np.savetxt('5-fold_sets/Encoded/Sets{}/X_train'.format(fold), train[:, :-1], delimiter=',')
    np.savetxt('5-fold_sets/Encoded/Sets{}/y_train'.format(fold), train[:, -1], delimiter=',')
    np.savetxt('5-fold_sets/Encoded/Sets{}/X_test'.format(fold), test[:, :-1], delimiter=',')
    np.savetxt('5-fold_sets/Encoded/Sets{}/y_test'.format(fold), test[:, -1], delimiter=',')

    #normalize continuos features
    norm = MaxAbsScaler()
    norm.fit(train[:, continuous])
    train[:, continuous] = norm.transform(train[:, continuous])
    test[:, continuous] = norm.transform(test[:, continuous])

    np.savetxt('5-fold_sets/Normalized/Sets{}/X_train'.format(fold), train[:, :-1], delimiter=',')
    np.savetxt('5-fold_sets/Normalized/Sets{}/y_train'.format(fold), train[:, -1], delimiter=',')
    np.savetxt('5-fold_sets/Normalized/Sets{}/X_test'.format(fold), test[:, :-1], delimiter=',')
    np.savetxt('5-fold_sets/Normalized/Sets{}/y_test'.format(fold), test[:, -1], delimiter=',')
    print

    #discretize continuos features
    disc = KBinsDiscretizer(n_bins=30, encode='ordinal', strategy='quantile')
    disc.fit(train[:, continuous])
    train[:, continuous] = disc.transform(train[:, continuous])
    test[:, continuous] = disc.transform(test[:, continuous])


    np.savetxt('5-fold_sets/Discretized/Sets{}/X_train'.format(fold), train[:, :-1], delimiter=',')
    np.savetxt('5-fold_sets/Discretized/Sets{}/y_train'.format(fold), train[:, -1], delimiter=',')
    np.savetxt('5-fold_sets/Discretized/Sets{}/X_test'.format(fold), test[:, :-1], delimiter=',')
    np.savetxt('5-fold_sets/Discretized/Sets{}/y_test'.format(fold), test[:, -1], delimiter=',')
