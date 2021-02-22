import sys
import numpy as np

# this script filters non-redundant samples from the dataset
# argv[1] = week 1-4 or 'all'
# argv[2] = class name 'normal' or 'attacker'

week_number = sys.argv[1]
class_name = sys.argv[2]

PATH = "WISENT-CIDDS-001/CIDDS-001/Reduced/OpenStack/"
FILE = "week{}.csv".format(week_number)

counter = 0
c2 = 0
unique = []
with open(PATH+FILE, "r") as fl:
    for line in fl:
        raw = fl.readline().split(",")[:-1]
        if len(raw) > 0:
            target = raw[-3]
            if class_name in target:
                c2 += 1
                a = tuple(np.array(raw)[[1,2,4,6,7,8,9,10]])
                if a not in unique:
                    unique.append(a)
                    counter += 1
                    with open(PATH+"new-{}-data-{}.csv".format(class_name,week_number), "a") as fl2:
                       for word in raw:
                           fl2.write(word.replace(' ', '') + ",")
                       fl2.write("\n")
