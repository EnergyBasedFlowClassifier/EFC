import numpy as np

INFO_DATA =[('Duration', 1, [(100,800000),(10,100),(1, 10),(0.04,1), (0.01,0.04), (0.006,0.01), (0.005,0.006), (0.004,0.005), (0.003,0.004), (0.002,0.003), (0.001,0.002), (0.000,0.001)]),
            ('Proto', 0),
            ('Src Pt', 1, [(60000, 70000), (40000, 60000), (500, 40000), (400,500), (100, 400), (60,100), (50,60), (0,50)]),
            ('Dst Pt', 1, [(60000, 70000), (40000, 60000), (500, 40000), (400,500), (100, 400), (60,100), (50,60), (0,50)]),
            ('Packets', 1, [(20, 1000000), (10, 20), (7,10), (6,7), (5,6), (4,5), (3,4), (2,3), (1,2)]),
            ('Bytes', 1, [(5000, 1000000000), (1000, 5000), (700, 1000), (500,700), (400,500), (300,400), (200,300), (110,200), (100,110), (90,100), (70,90), (60,70), (50,60), (0,50)]),
            ('Flags', 0),
            ('Tos', 0)]

PROTOCOLS = {"ICMP":29, "TCP":31, "UDP":30, "IGMP":28, "GRE":27}

FLAGS_DICT = {"......":31, ".....F":1, "....S.":2, "...R..":3, "..P...":4, ".A....":5,
              "....SF":6, "...R.F":7, "..P..F":8, ".A...F":9, "...RS.":10, "..P.S.":11,
              ".A..S.":12, "..PR..":13, ".A.R..":14, ".AP...":15, "...RSF":16,
              "..P.SF":17, ".A..SF":18, "..PR.F":19, ".A.R.F":20, ".AP..F":21,
              "..PRS.":22, ".A.RS.":23, ".AP.S.":24, ".APR..":25, "..PRSF":26,
              ".A.RSF":27, ".AP.SF":28, ".APR.F":29, ".APRS.":30, ".APRSF":0,
              "0xdb":28, "0xc2":2, "0xda":24, "0xdf":31, "0xd7":27, "0xd6":23,
              "0xd2":12, "0x53":18, "0x5b":28, "0x52":12, "0x5a":24, "0xd3":18 }

TOS_DICT = {"0":31, "16":30, "32":29, "192":28}

def discretize_features(data, n_bins):
    for k in range(data.shape[1]):
        unique, counts = np.unique(data[:,k], return_counts=True)
        data_k = data[:,k]
        if INFO_DATA[k][1]:
            data_k = []
            for item in data[:,k]:
                if "M" in item:
                    data_k.append(float(item.replace('M',''))*1000000)
                elif "K" in item:
                    data_k.append(float(item.replace('K',''))*1000)
                else:
                    data_k.append(float(item))
            if len(unique) > 1:
                new = []
                for item in data_k:
                    for i, rng in enumerate(INFO_DATA[k][2]):
                        if float(item) >= rng[0] and float(item) < rng[1]:
                            new.append(i)

                unique, counts = np.unique(new, return_counts=True)

                data[:,k] = new
        else:
            if INFO_DATA[k][0] == "Proto":
                for i, item in enumerate(data_k):
                    data_k[i] = PROTOCOLS[item.replace(" ","")]
            elif INFO_DATA[k][0] == "Flags":
                for i, item in enumerate(data_k):
                    data_k[i] = FLAGS_DICT[item.replace(" ","")]
            else:
                for i, item in enumerate(data_k):
                    data_k[i] = TOS_DICT[item.replace(" ","")]
            unique, counts = np.unique(data_k, return_counts=True)
            data[:,k] = data_k
    return data

def select_features(data, attack_types):
    # attack_types = lista dos tipos do ataque de interesse

    normal = np.array([x[:-1] for x in data if x[-1] == "normal"])
    attack = np.array([x[:-1] for x in data if x[-1] in attack_types])

    selected_idxs = []
    for k in range(normal.shape[1]):
        unique_n, counts_n = np.unique(normal[:,k], return_counts=True)
        unique_a, counts_a = np.unique(attack[:,k], return_counts=True)
        if (len(unique_a) > 1) or (len(unique_n) > 1):
            selected_idxs.append(k)
    selected_normal = np.array(normal[:,selected_idxs],dtype=int)
    selected_attack = np.array(attack[:,selected_idxs],dtype=int)

    return selected_normal, selected_attack, selected_idxs
