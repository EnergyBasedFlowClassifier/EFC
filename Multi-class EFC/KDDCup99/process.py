from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, KBinsDiscretizer, Normalizer, StandardScaler, MaxAbsScaler
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

malicious_names = [['normal.'], ['back.', 'smurf.', 'teardrop.', 'neptune.', 'land.', 'pod.'],
['ipsweep.',  'portsweep.',  'satan.',  'nmap.'], ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.',
 'warezclient.', 'warezmaster.', 'phf.'], ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']]

unknown_names = ['apache2.', 'mailbomb.', 'udpstorm.', 'processtable.','mscan.', 'saint.', 'named.',
'sendmail.', 'snmpgetattack.', 'snmpguess.', 'worm.', 'xlock.', 'xsnoop.',
'ps.', 'sqlattack.', 'xterm.', 'httptunnel.']


#group continuos and symbolic features indexes
symbolic = [1,2,3,6,11,20,21]
continuous = [x for x in range(41) if x not in symbolic]

# load data
train = np.array(pd.read_csv('Data/train', header=None))
validation = np.array(pd.read_csv('Data/validation', header=None))
test = pd.read_csv('Data/corrected', header=None)

indexes = np.where(np.isin(test.iloc[:,-1], unknown_names))[0]
np.save("unknown_idx", indexes)

test_unknown = test.iloc[indexes, :]
aux = test_unknown.duplicated()
dup_index = aux[aux==True].index

test_unknown = test.drop(dup_index, axis=0)
indexes = np.where(np.isin(test_unknown.iloc[:,-1], unknown_names))[0]
np.save("unknown_unique_idx", indexes)


test = np.array(test)
test_unknown = np.array(test_unknown)

#encode symbolic features
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
enc.fit(train[:, symbolic])
train[:, symbolic] = enc.transform(train[:, symbolic])
validation[:, symbolic] = enc.transform(validation[:, symbolic])
test[:, symbolic] = enc.transform(test[:, symbolic])
test[:, symbolic] = np.nan_to_num(test[:, symbolic].astype('float'), nan=np.max(test[:, symbolic])+1)
test_unknown[:, symbolic] = enc.transform(test_unknown[:, symbolic])
test_unknown[:, symbolic] = np.nan_to_num(test_unknown[:, symbolic].astype('float'), nan=np.max(test_unknown[:, symbolic])+1)



#encode labels
for idx in range(len(malicious_names)):
    train[:, -1] = [idx if value in malicious_names[idx] else value for value in train[:,-1]]
    validation[:, -1] = [idx if value in malicious_names[idx] else value for value in validation[:,-1]]
    test[:, -1] = [idx if value in malicious_names[idx] else value for value in test[:,-1]]
    test_unknown[:, -1] = [idx if value in malicious_names[idx] else value for value in test_unknown[:,-1]]
test[:, -1] = [100 if value in unknown_names else value for value in test[:,-1]]
test_unknown[:, -1] = [100 if value in unknown_names else value for value in test_unknown[:,-1]]



#normalize continuos features
norm = MaxAbsScaler()
norm.fit(train[:, continuous])
train[:, continuous] = norm.transform(train[:, continuous])
validation[:, continuous] = norm.transform(validation[:, continuous])
test[:, continuous] = norm.transform(test[:, continuous])
test_unknown[:, continuous] = norm.transform(test_unknown[:, continuous])

np.savetxt('Data/Normalized/X_train', train[:, :-1], delimiter=',')
np.savetxt('Data/Normalized/y_train', train[:, -1], delimiter=',')

np.savetxt('Data/Normalized/X_validation', validation[:, :-1], delimiter=',')
np.savetxt('Data/Normalized/y_validation', validation[:, -1], delimiter=',')

np.savetxt('Data/Normalized/X_test', test[:, :-1], delimiter=',')
np.savetxt('Data/Normalized/y_test', test[:, -1], delimiter=',')

np.savetxt('Data/Normalized/X_test_unknown_unique', test_unknown[:, :-1], delimiter=',')
np.savetxt('Data/Normalized/y_test_unknown_unique', test_unknown[:, -1], delimiter=',')

#discretize continuos features
disc = KBinsDiscretizer(n_bins=21, encode='ordinal', strategy='quantile')
disc.fit(train[:, continuous])
train[:, continuous] = disc.transform(train[:, continuous])
validation[:, continuous] = disc.transform(validation[:, continuous])
test[:, continuous] = disc.transform(test[:, continuous])
test_unknown[:, continuous] = disc.transform(test_unknown[:, continuous])

np.savetxt('Data/Normalized-Discretized/X_train', train[:, :-1], delimiter=',')
np.savetxt('Data/Normalized-Discretized/y_train', train[:, -1], delimiter=',')

np.savetxt('Data/Normalized-Discretized/X_validation', validation[:, :-1], delimiter=',')
np.savetxt('Data/Normalized-Discretized/y_validation', validation[:, -1], delimiter=',')

np.savetxt('Data/Normalized-Discretized/X_test', test[:, :-1], delimiter=',')
np.savetxt('Data/Normalized-Discretized/y_test', test[:, -1], delimiter=',')

np.savetxt('Data/Normalized-Discretized/X_test_unknown_unique', test_unknown[:, :-1], delimiter=',')
np.savetxt('Data/Normalized-Discretized/y_test_unknown_unique', test_unknown[:, -1], delimiter=',')
