import sys
import numpy as np
import random as rd
import os
from discretize import *

# this script creates the training and test sets to perform cross validation
# it creates the same sets to evaluate efc and classical ml methods
# argv[1] can be 'os' or 'ext'


os.makedirs("CIDDS-001/test_sets", exist_ok=True)
os.makedirs("CIDDS-001/test_sets/non-discretized", exist_ok=True)
os.makedirs("CIDDS-001/test_sets/discretized", exist_ok=True)

TRAFFIC = sys.argv[1]           # openstack traffic 'os' for openstack or 'ext' for external traffic
REP = sys.argv[2]               # test set number
TRAINING_4_DNN = 1              # write continuous datasets?
OUT_PATH = "CIDDS-001/test_sets/"
PATH = "CIDDS-001/Reduced/OpenStack/"
PATH_EXT =  "CIDDS-001/Reduced/ExternalServer/"
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
N_TRAIN = 10000
N_TEST = 10000
Q = 32          # alphabet size

normal_all = []
if TRAFFIC == "ext":
    with open(PATH_EXT +"new-unknown-data-all.csv", "r") as fl:
        raw = [x for x in fl.read().split("\n")[:-1]]
        normal_all += [x.split(",")[:-1] for x in raw]
else:
    with open(PATH +"new-normal-data-all.csv", "r") as fl:
        raw = [x for x in fl.read().split("\n")[:-1]]
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
if TRAFFIC != "ext":
    atk_file = "new-attacker-data-all.csv"

    with open(PATH+atk_file, "r") as fl:
        raw = [x for x in fl.read().split("\n")[:-1] if len(x.split(",")) > 1]

    data_atk = [x.split(",")[:-1] for x in raw]

else:
    atk_file = "new-suspicious-data-all.csv"

    with open(PATH_EXT+atk_file, "r") as fl:
        raw = [x for x in fl.read().split("\n")[:-1] if len(x.split(",")) > 1]

    data_atk = [x.split(",")[:-1] for x in raw]

rd.shuffle(data_atk)

labels_atk = []
if TRAFFIC != "ext":
    classes = ["bruteForce", "pingScan", "portScan", "dos"]
else:
    classes = ["suspicious"]
n_samples = int((N_TEST + N_TRAIN)/2)
if TRAFFIC != "ext":
    print(data_atk[1])
    samples_atk = rd.sample([x for x in data_atk if x[-2] == "pingScan"],100)
    print(len(samples_atk))
    samples_atk += rd.sample([x for x in data_atk if x[-2] == "bruteForce"],150)
    print(len(samples_atk))
    samples_atk += rd.sample([x for x in data_atk if x[-2] == "portScan"],800)
    print(len(samples_atk))
    n_remaining = N_TEST - len(samples_atk)
    samples_atk += rd.sample([x for x in data_atk if x[-2] == "dos"],n_remaining)
    print(len(samples_atk))
else:
    samples_atk = rd.sample(data_atk,int(n_samples))
if TRAFFIC != "ext":
    labels_atk = [x[-2] for x in samples_atk]
else:
    labels_atk = [x[-3] for x in samples_atk]
for c in classes:
    print(c, len([x for x in samples_atk if x[-2] == c]))

if TRAFFIC != "ext":
    labels = ["normal"]*N_TEST + labels_atk
else:
    labels = ["unknown"]*N_TEST + labels_atk

new_data = []
for item in samples_atk:
    new_item = []
    for i, elem in enumerate(item):
        if LABELS[i][1]:
            new_item.append(elem)
    new_data.append(new_item)
data_atk = new_data.copy()

if TRAINING_4_DNN:
    dnn_data = []
    for item in data_normal + data_atk:
        new_item = []
        for i, elem in enumerate(item):
            if i in [0,2,3,4,5,7]:
                if "M" in elem:
                    new_item.append(float(elem.replace('M', ''))*1000000)
                elif "K" in item:
                    new_item.append(float(elem.replace('K', ''))*1000)
                else:
                    new_item.append(float(elem))
            elif i == 1:
                new_item.append(PROTOCOLS[elem.replace(" ", "")])
            else:
                new_item.append(FLAGS_DICT[elem.replace(" ", "")])
        dnn_data.append(new_item)

    with open(OUT_PATH+"non-discretized/{}_test_cidds_{}.csv".format(REP,TRAFFIC), "w") as fl:
        for i, item in enumerate(dnn_data):
            if i < N_TEST:
                fl.write(labels[i]+",")
            else:
                fl.write(labels[i]+",")
            for j,elem in enumerate(item):
                if j != 7:
                    fl.write(str(elem) + ",")
                else:
                    fl.write(str(elem) + "\n")

all_atk = discretize_features(np.array(data_atk), Q)
all_normal = discretize_features(np.array(data_normal), Q)

np.save(OUT_PATH+"discretized/{}_test_{}".format(REP,TRAFFIC),np.array(list(all_normal)+list(all_atk)))
np.save(OUT_PATH+"discretized/{}_labels_{}".format(REP,TRAFFIC), labels)
