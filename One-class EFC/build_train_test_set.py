import numpy as np
import pandas as pd
import random

#this script creates train and test sets, for each file used in the experiments, to be used in the domain adaptation
#experiment

def build_train_test_friday():
    data = np.load("Discretized_unique/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.npy", allow_pickle=True)
    normals = [x for x in data if x[-1] == 1]
    atack = [x for x in data if x[-1] == 2]
    normal_samples = random.Random(20).sample(normals, 1250+1250)
    atack_samples = random.Random(20).sample(atack, 1250+1250)

    train_normal = [x[1:-1] for x in normal_samples[:1250]]
    train_normal_idx = [x[0] for x in normal_samples[:1250]]
    np.save("External_test/Discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Train_normal.npy", train_normal)

    train_malicious = [x[1:-1] for x in atack_samples[:1250]]
    train_malicious_idx = [x[0] for x in atack_samples[:1250]]
    np.save("External_test/Discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Train_malicious.npy", train_malicious)

    test = [x[1:-1] for x in normal_samples[1250::] + atack_samples[1250::]]
    np.save("External_test/Discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Test.npy", test)

    test_idx = [x[0] for x in normal_samples[1250::] + atack_samples[1250::]]
    test_labels = [x[-1] for x in normal_samples[1250::] + atack_samples[1250::]]
    np.save("External_test/Discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Test_labels.npy", test_labels)


    data = np.array(pd.read_csv("Pre_processed/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"))

    train_normal = data[[x for x in train_normal_idx], 1:-1]
    np.save("External_test/Non_discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Train_normal.npy", train_normal)

    train_normal_labels = data[[x for x in train_normal_idx], -1]
    np.save("External_test/Non_discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Train_normal_labels.npy", train_normal_labels)

    train_malicious = data[[x for x in train_malicious_idx], 1:-1]
    np.save("External_test/Non_discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Train_malicious.npy", train_malicious)

    train_malicious_labels = data[[x for x in train_malicious_idx], -1]
    np.save("External_test/Non_discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Train_malicious_labels.npy", train_malicious_labels)

    test = data[[x for x in test_idx], 1:-1]
    np.save("External_test/Non_discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Test.npy", test)

    test_labels = data[[x for x in test_idx], -1]
    np.save("External_test/Non_discretized/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX/Test_labels.npy", test_labels)

def build_train_test_wednesday():
    data = np.load("Discretized_unique/Wednesday-workingHours.pcap_ISCX.npy", allow_pickle=True)
    normals = [x for x in data if x[-1] == 1]
    hulk_atacks = [x for x in data if x[-1] == 11]
    golden_atacks = [x for x in data if x[-1] == 12]
    slowloris_atacks = [x for x in data if x[-1] == 13]
    slowhttp_atacks = [x for x in data if x[-1] == 14]

    normal_samples = random.Random(20).sample(normals, 1250+1250)
    hulk_samples = [x for x in random.Random(20).sample(hulk_atacks, 850)]
    golden_samples = [x for x in random.Random(20).sample(golden_atacks, 850)]
    slowloris_samples = [x for x in random.Random(20).sample(slowloris_atacks, 370)]
    slowhttp_samples = [x for x in random.Random(20).sample(slowhttp_atacks, 430)]


    train_normal = [x[1:-1] for x in normal_samples[:1250]]
    train_normal_idx = [x[0] for x in normal_samples[:1250]]
    np.save("External_test/Discretized/Wednesday-workingHours.pcap_ISCX/Train_normal.npy", train_normal)

    train_malicious = [x[1:-1] for x in hulk_samples[:425] + golden_samples[:425] + slowloris_samples[:185] + slowhttp_samples[:215]]
    train_malicious_idx = [x[0] for x in hulk_samples[:425] + golden_samples[:425] + slowloris_samples[:185] + slowhttp_samples[:215]]
    np.save("External_test/Discretized/Wednesday-workingHours.pcap_ISCX/Train_malicious.npy", train_malicious)

    test = [x[1:-1] for x in normal_samples[1250::] + hulk_samples[425::]+ golden_samples[425::] + slowloris_samples[185::] + slowhttp_samples[215::]]
    np.save("External_test/Discretized/Wednesday-workingHours.pcap_ISCX/Test.npy", test)

    test_idx = [x[0] for x in normal_samples[1250::] + hulk_samples[425::]+ golden_samples[425::] + slowloris_samples[185::] + slowhttp_samples[215::]]
    test_labels = [x[-1] for x in normal_samples[1250::] + hulk_samples[425::]+ golden_samples[425::] + slowloris_samples[185::] + slowhttp_samples[215::]]
    np.save("External_test/Discretized/Wednesday-workingHours.pcap_ISCX/Test_labels.npy", test_labels)


    data = np.array(pd.read_csv("Pre_processed/Wednesday-workingHours.pcap_ISCX.csv"))

    train_normal = data[[x for x in train_normal_idx], 1:-1]
    np.save("External_test/Non_discretized/Wednesday-workingHours.pcap_ISCX/Train_normal.npy", train_normal)

    train_normal_labels = data[[x for x in train_normal_idx], -1]
    np.save("External_test/Non_discretized/Wednesday-workingHours.pcap_ISCX/Train_normal_labels.npy", train_normal_labels)

    train_malicious = data[[x for x in train_malicious_idx], 1:-1]
    np.save("External_test/Non_discretized/Wednesday-workingHours.pcap_ISCX/Train_malicious.npy", train_malicious)

    train_malicious_labels = data[[x for x in train_malicious_idx], -1]
    np.save("External_test/Non_discretized/Wednesday-workingHours.pcap_ISCX/Train_malicious_labels.npy", train_malicious_labels)

    test = data[[x for x in test_idx], 1:-1]
    np.save("External_test/Non_discretized/Wednesday-workingHours.pcap_ISCX/Test.npy", test)

    test_labels = data[[x for x in test_idx], -1]
    np.save("External_test/Non_discretized/Wednesday-workingHours.pcap_ISCX/Test_labels.npy", test_labels)


def build_train_test_joined_files():
    data = np.load("Discretized_unique/Joined_files.npy", allow_pickle=True)
    normals = [x for x in data if x[-1] == 1]
    portscan_atacks = [x for x in data if x[-1] == 3]
    bot_atacks = [x for x in data if x[-1] == 4]
    inf_atacks = [x for x in data if x[-1] == 5]
    webBF_atacks = [x for x in data if x[-1] == 6]
    webSQL_atacks = [x for x in data if x[-1] == 7]
    webXSS_atacks = [x for x in data if x[-1] == 8]
    FTP_atacks = [x for x in data if x[-1] == 9]
    SSH_atacks = [x for x in data if x[-1] == 10]

    normal_samples = random.Random(20).sample(normals, 1250+1250)
    portscan_samples = [x for x in random.Random(20).sample(portscan_atacks, 1460)]
    bot_samples = [x for x in random.Random(20).sample(bot_atacks, 840)]
    inf_samples = [x for x in random.Random(20).sample(inf_atacks, 64)]
    webBF_samples = [x for x in random.Random(20).sample(webBF_atacks, 516)]
    webSQL_samples = [x for x in random.Random(20).sample(webSQL_atacks, 36)]
    webXSS_samples = [x for x in random.Random(20).sample(webXSS_atacks, 148)]
    FTP_samples = [x for x in random.Random(20).sample(FTP_atacks, 1460)]
    SSH_samples = [x for x in random.Random(20).sample(SSH_atacks, 476)]


    train_normal = [x[1:-1] for x in normal_samples[:1250]]
    train_normal_idx = [x[0] for x in normal_samples[:1250]]
    np.save("External_test/Discretized/Joined_files/Train_normal.npy", train_normal)

    train_malicious = [x[1:-1] for x in portscan_samples[:730] + bot_samples[:420] + inf_samples[:32] + webBF_samples[:258] + webSQL_samples[:18] +
    webXSS_samples[:74] + FTP_samples[:730] + SSH_samples[:238]]
    train_malicious_idx = [x[0] for x in portscan_samples[:730] + bot_samples[:420] + inf_samples[:32] + webBF_samples[:258] + webSQL_samples[:18] +
    webXSS_samples[:74] + FTP_samples[:730] + SSH_samples[:238]]
    np.save("External_test/Discretized/Joined_files/Train_malicious.npy", train_malicious)

    test = [x[1:-1] for x in normal_samples[1250::] + portscan_samples[730::]+ bot_samples[420::] + inf_samples[32::] + webBF_samples[258::] + webSQL_samples[18::] +
    webXSS_samples[74::] + FTP_samples[730::] + SSH_samples[238::]]
    np.save("External_test/Discretized/Joined_files/Test.npy", test)

    test_idx = [x[0] for x in normal_samples[1250::] + portscan_samples[730::]+ bot_samples[420::] + inf_samples[32::] + webBF_samples[258::] + webSQL_samples[18::] +
    webXSS_samples[74::] + FTP_samples[730::] + SSH_samples[238::]]
    test_labels = [x[-1] for x in normal_samples[1250::] + portscan_samples[730::]+ bot_samples[420::] + inf_samples[32::] + webBF_samples[258::] + webSQL_samples[18::] +
    webXSS_samples[74::] + FTP_samples[730::] + SSH_samples[238::]]
    np.save("External_test/Discretized/Joined_files/Test_labels.npy", test_labels)


    data = np.array(pd.read_csv("Pre_processed/Joined_files.csv"))

    train_normal = data[[x for x in train_normal_idx], 1:-1]
    np.save("External_test/Non_discretized/Joined_files/Train_normal.npy", train_normal)

    train_normal_labels = data[[x for x in train_normal_idx], -1]
    np.save("External_test/Non_discretized/Joined_files/Train_normal_labels.npy", train_normal_labels)

    train_malicious = data[[x for x in train_malicious_idx], 1:-1]
    np.save("External_test/Non_discretized/Joined_files/Train_malicious.npy", train_malicious)

    train_malicious_labels = data[[x for x in train_malicious_idx], -1]
    np.save("External_test/Non_discretized/Joined_files/Train_malicious_labels.npy", train_malicious_labels)

    test = data[[x for x in test_idx], 1:-1]
    np.save("External_test/Non_discretized/Joined_files/Test.npy", test)

    test_labels = data[[x for x in test_idx], -1]
    np.save("External_test/Non_discretized/Joined_files/Test_labels.npy", test_labels)

build_train_test_wednesday()
build_train_test_friday()
build_train_test_joined_files()
