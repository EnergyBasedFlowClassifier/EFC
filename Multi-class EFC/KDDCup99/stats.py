import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, KBinsDiscretizer, Normalizer, StandardScaler, MaxAbsScaler

test = pd.read_csv('Data/corrected', header=None)
train = np.array(pd.read_csv('Data/train', header=None))
symbolic = [1,2,3,6,11,20,21] #

unknown_list = [['apache2.', 'mailbomb.', 'udpstorm.', 'processtable.'],
['mscan.', 'saint.'], ['named.', 'sendmail.', 'snmpgetattack.', 'snmpguess.', 'worm.', 'xlock.', 'xsnoop.'],
 ['ps.', 'sqlattack.', 'xterm.', 'httptunnel.']]

indexes =[]
for attack_list in unknown_list:
    for label in attack_list:
        indexes = list(np.where(test.iloc[:,-1]==label)[0])
        selected = test.iloc[indexes, :]
        print(label, selected.shape[0], selected.drop_duplicates().shape[0])
        input()

#to discover the detection rates for each unknown label
all_ocurrences =[]
for attack_list in unknown_list:
    class_ocurrences = []
    for label in attack_list:
        class_ocurrences.append(np.where(test[:,-1]==label)[0])
    all_ocurrences.append(class_ocurrences)
print(all_ocurrences[0][0])
np.save("unknown_idx", all_ocurrences)


# enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
# enc.fit(train[:, symbolic])
# train[:, symbolic] = enc.transform(train[:, symbolic])
# test[:, symbolic] = enc.transform(test[:, symbolic])
# test[:, symbolic] = np.nan_to_num(test[:, symbolic].astype('float'), nan=np.max(test[:, symbolic])+1)
#
# with open("unknown_stats", "w+") as file:
#     unique = np.unique(test[indexes, :-1].astype('float'), axis=0)
#     print(unique.shape[0])
#     count = 0
#     for row in unique:
#         count += 1
#         row_equals = (test[:, :-1].astype('float') == row).all(axis=1).nonzero()
#         unique2, counts = np.unique(test[row_equals, -1], return_counts=True)
#         if unique2.shape[0] > 1:
#             file.write("{} {}".format(unique2, counts))
