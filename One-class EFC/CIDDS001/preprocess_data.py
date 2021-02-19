import sys
import numpy as np

# this script filters non-redundant samples from the dataset
# argv[1] = week (1-4)
# argv[2] = class name (normal  dos, pingscan, portscan, bruteforce)
# obs:  if class name is normal, it should be raw[-3] on line 18
#       if class name is dos, pingscan, portscan, bruteforce, it should be raw[-2] on line 18

PATH = "data/CIDDS-001/traffic/OpenStack/"
FILE = "CIDDS-001-internal-week{}.csv".format(sys.argv[1])

counter = 0
c2 = 0
unique = []
with open(PATH+FILE, "r") as fl:
    for line in fl:
        raw = fl.readline().split(",")[:-1]
        if len(raw) > 0 and sys.argv[2] in raw[-2]:
            c2 += 1
            a = tuple(np.array(raw)[[1,2,4,6,7,8,9,10]]) 
            if a not in unique:
                unique.append(a)
                counter += 1
                with open(PATH+"new-{}-data-{}.csv".format(sys.argv[2],sys.argv[1]), "a") as fl2:
                    for word in raw:
                        fl2.write(word + ", ")
                    fl2.write("\n")
                    print(counter, c2)

