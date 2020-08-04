import sys
import numpy as np

# argv[1] can be 1,2,3 or 4
# argv[2] can be unknown or suspicious

PATH = "../data/CIDDS-001/traffic/ExternalServer/"
FILE = "CIDDS-001-external-week{}.csv".format(sys.argv[1])

counter = 0
c2 = 0
unique = []
with open(PATH+FILE, "r") as fl:
    for line in fl:
        raw = fl.readline().split(",")[:-1]
        if len(raw) > 0 and sys.argv[2] in raw[-3]:
            c2 += 1
            a = tuple(np.array(raw)[[1,2,4,6,7,8,9,10]])
            if a not in unique:
                unique.append(a)
                with open(PATH+"new-{}-data-{}.csv".format(sys.argv[2],sys.argv[1]), "a") as fl2:
                    for word in raw:
                        fl2.write(word + ", ")
                    fl2.write("\n")
                    counter += 1
                    print(counter, c2)

