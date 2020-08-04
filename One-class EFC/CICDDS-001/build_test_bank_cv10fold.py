import sys
import numpy as np
import random as rd
from classification_functions import *

# this script creates the training and test sets to perform cross validation
# it creates the same sets to evaluate efc and classical ml methods
# argv[1] can be 1, 2, 3, 4 or ext
# argv[2] can be 1, 2, 3 or 4

OS_WEEK = sys.argv[1]           # openstack traffic week or 'ext' to indicate external traffic
EXT_WEEK = sys.argv[2]          # external traffic week
TRAINING_4_DNN = 1              # write continuous datasets?
PATH = "../data/CIDDS-001/traffic/OpenStack/"
PATH_EXT =  "../data/CIDDS-001/traffic/ExternalServer/"
TEST_FILE = "new-attack-data-{}.csv"
LABELS = [  ('Date first seen', 0, 0),
            ('Duration', 1, 1),
            ('Proto', 1, 0),
            ('Src IP Addr', 0, 0),
            ('Src Pt', 1, 1),
            ('Dst IP Addr', 0, 0),
            ('Dst Pt', 1, 1),
            ('Packets', 1, 1),
            ('Bytes', 1, 1),
            ('Flows', 0, 1),
            ('Flags', 1, 0),
            ('Tos', 1, 1),
            ('class', 0, 0),
            ('attackType', 0, 0),
            ('attackID', 0, 0),
            ('attackDescription', 0, 0)] # name, used, must_digitize
R = 1
N_TRAIN = int(9000/R)
N_TEST = int(1000/R)
Q = 32

normal_all = []
if sys.argv[1] == "ext":
    with open(PATH_EXT +"new-unknown-data-{}.csv".format(EXT_WEEK), "r") as fl:
        raw = [x for x in fl.read().split("\n")[:-1]]
        normal_all += [x.split(",")[:-1] for x in raw]
else:
    with open(PATH +"new-normal-data-{}.csv".format(OS_WEEK), "r") as fl:
        raw = [x for x in fl.read().split("\n")[:-1]][:100000]
        normal_all += [x.split(",")[:-1] for x in raw]
rd.shuffle(normal_all)

if len(normal_all) >= int(N_TRAIN/2) + int(N_TEST/2): 
    samples_normal = rd.sample(normal_all, int(N_TRAIN/2) + int(N_TEST/2))
else:
    samples_normal = rd.sample(normal_all+normal_all, int(N_TRAIN/2) + int(N_TEST/2))

data_normal = []
for item in samples_normal:
    new_item = []
    for i, elem in enumerate(item):
        if LABELS[i][1]:
            new_item.append(elem)
    data_normal.append(new_item)

data_atk = []
if sys.argv[1] != "ext":
    atk_file = "new-attacker-data-{}.csv".format(OS_WEEK)

    with open(PATH+atk_file, "r") as fl:
        raw = [x for x in fl.read().split("\n")[:-1] if len(x.split(",")) > 1 and "attacker" in x.split(",")[-4]]

    data_atk = [x.split(",")[:-1] for x in raw]
        
else:
    atk_file = "new-suspicious-data-{}.csv".format(EXT_WEEK)

    with open(PATH_EXT+atk_file, "r") as fl:
        raw = [x for x in fl.read().split("\n")[:-1] if len(x.split(",")) > 1]

    data_atk = [x.split(",")[:-1] for x in raw]

rd.shuffle(data_atk)

samples_atk = []
if sys.argv[1] != "ext":
    classes = ["bruteForce", "pingScan", "portScan", "dos"]
else:
    classes = ["suspicious"]
n_samples = int((N_TEST + N_TRAIN)/2)
for atk in classes:
    all_atk = [x for x in data_atk if atk in x[-3]]
    if len(all_atk) < n_samples:
        print(len(all_atk))
        n_samples -= len(all_atk)
        samples_atk += all_atk
    else:
        samples_atk += rd.sample(all_atk,int(n_samples))
        print(n_samples)

new_data = []
for item in samples_atk:
    new_item = []
    for i, elem in enumerate(item):
        if LABELS[i][1]:
            new_item.append(elem)
    new_data.append(new_item)
data_atk = new_data.copy()

new_normal_data = discretize_features(np.array(data_normal), Q)
new_malicious_data = discretize_features(np.array(data_atk), Q)

normal_selected = []
idxs_normal = []
for i in range(len(new_normal_data)):
    x = list(new_normal_data[i,:])
    if x not in normal_selected:
        normal_selected.append(x)
        idxs_normal.append(i)
print(len(normal_selected))

malicious_selected = []
idxs_malicious = []
for i in range(len(new_malicious_data)):
    x = list(new_malicious_data[i,:])
    if x not in malicious_selected:
        malicious_selected.append(x)
        idxs_malicious.append(i)
print(len(malicious_selected))

N_SELECTED = int(np.min([len(malicious_selected), len(normal_selected)])/2)

for cv in range(10):
    FOLD_M = int(125/R)
    FOLD_N = int(500/R)
    test_normal = data_normal[cv*FOLD_N:(cv+1)*FOLD_N]
    train_normal = data_normal[:cv*FOLD_N] + data_normal[(cv+1)*FOLD_N:]
    test_malicious = []
    train_malicious = []
    for k in range(4):
        data = data_atk[k*FOLD_M*10:(k+1)*FOLD_M*10]
        test_malicious += data[cv*FOLD_M:(cv+1)*FOLD_M]
        train_malicious += data[:cv*FOLD_M] + data[(cv+1)*FOLD_M:]
    print(len(test_malicious), len(train_malicious), len(test_normal), len(train_normal))

    if TRAINING_4_DNN:
        dnn_data = []
        for item in test_normal + test_malicious + train_normal + train_malicious:
            new_item = []
            for i, elem in enumerate(item):
                if i in [0,2,3,4,5,7]:
                    if "M" in elem:
                        new_item.append(float(elem.split(" ")[-2])*1000000)
                    elif "K" in item:
                        new_item.append(float(elem.split(" ")[-2])*1000)
                    else:
                        new_item.append(float(elem))
                elif i == 1:
                    new_item.append(PROTOCOLS[elem.replace(" ", "")])
                else:
                    new_item.append(FLAGS_DICT[elem.replace(" ", "")])
            dnn_data.append(new_item)
        dnn_test_data = dnn_data[:FOLD_N*2]
        dnn_train_data = dnn_data[FOLD_N*2:]

        with open("dnn/cv{}.testing_cidds_random{}_w{}.csv".format(cv,EXT_WEEK,sys.argv[1]), "w") as fl:
            for i, item in enumerate(dnn_test_data):
                if i < FOLD_N:
                    fl.write("1,")
                else:
                    fl.write("0,")
                for j,elem in enumerate(item):
                    if j != 7:
                        fl.write(str(elem) + ",")
                    else:
                        fl.write(str(elem) + "\n")

        with open("dnn/cv{}.training_cidds_random{}_w{}.csv".format(cv,EXT_WEEK,sys.argv[1]), "w") as fl:
            for i, item in enumerate(dnn_train_data):
                if i < FOLD_N*9:
                    fl.write("1,")
                else:
                    fl.write("0,")
                for j,elem in enumerate(item):
                    if j != 7:
                        fl.write(str(elem) + ",")
                    else:
                        fl.write(str(elem) + "\n")

        dnn_selected_normal = []
        for idx, item in enumerate(data_normal):
            if idx in idxs_normal:
                new_item = []
                for i, elem in enumerate(item):
                    if i in [0,2,3,4,5,7]:
                        if "M" in elem:
                            new_item.append(float(elem.split(" ")[-2])*1000000)
                        elif "K" in item:
                            new_item.append(float(elem.split(" ")[-2])*1000)
                        else:
                            new_item.append(float(elem))
                    elif i == 1:
                        new_item.append(PROTOCOLS[elem.replace(" ", "")])
                    else:
                        new_item.append(FLAGS_DICT[elem.replace(" ", "")])
                dnn_selected_normal.append(new_item)

        dnn_selected_malicious = []
        for idx, item in enumerate(data_atk):
            if idx in idxs_malicious:
                new_item = []
                for i, elem in enumerate(item):
                    if i in [0,2,3,4,5,7]:
                        if "M" in elem:
                            new_item.append(float(elem.split(" ")[-2])*1000000)
                        elif "K" in item:
                            new_item.append(float(elem.split(" ")[-2])*1000)
                        else:
                            new_item.append(float(elem))
                    elif i == 1:
                        new_item.append(PROTOCOLS[elem.replace(" ", "")])
                    else:
                        new_item.append(FLAGS_DICT[elem.replace(" ", "")])
                dnn_selected_malicious.append(new_item)

        with open("dnn/cv{}.testing_cidds_selected{}_w{}.csv".format(cv,EXT_WEEK,sys.argv[1]), "w") as fl:
            for i, item in enumerate(dnn_selected_normal[:N_SELECTED] + dnn_selected_malicious[:N_SELECTED]):
                if i < N_SELECTED:
                    fl.write("1,")
                else:
                    fl.write("0,")
                for j,elem in enumerate(item):
                    if j != 7:
                        fl.write(str(elem) + ",")
                    else:
                        fl.write(str(elem) + "\n")

        with open("dnn/cv{}.training_cidds_selected{}_w{}.csv".format(cv,EXT_WEEK,sys.argv[1]), "w") as fl:
            for i, item in enumerate(dnn_selected_normal[N_SELECTED:2*N_SELECTED] + dnn_selected_malicious[N_SELECTED:2*N_SELECTED]):
                if i < N_SELECTED:
                    fl.write("1,")
                else:
                    fl.write("0,")
                for j,elem in enumerate(item):
                    if j != 7:
                        fl.write(str(elem) + ",")
                    else:
                        fl.write(str(elem) + "\n")

    new_test_malicious = discretize_features(np.array(test_malicious), Q)
    new_test_normal = discretize_features(np.array(test_normal), Q)
    new_train_normal = discretize_features(np.array(train_normal), Q)
    new_train_malicious = discretize_features(np.array(train_malicious), Q)

    OUT_PATH = "training_test_sets/"
    np.save(OUT_PATH+"cv{}.test_random{}_w{}".format(cv,EXT_WEEK,sys.argv[1]),np.array(list(new_test_normal) + list(new_test_malicious),dtype=int))
    np.save(OUT_PATH+"cv{}.train_random{}_w{}".format(cv,EXT_WEEK,sys.argv[1]),np.array(new_train_normal,dtype=int))
    np.save(OUT_PATH+"cv{}.test_selected{}_w{}".format(cv,EXT_WEEK,sys.argv[1]),np.array(list(normal_selected)[:N_SELECTED] + list(malicious_selected)[:N_SELECTED],dtype=int))
    np.save(OUT_PATH+"cv{}.train_selected{}_w{}".format(cv,EXT_WEEK,sys.argv[1]),np.array(list(normal_selected)[N_SELECTED:2*N_SELECTED],dtype=int))
