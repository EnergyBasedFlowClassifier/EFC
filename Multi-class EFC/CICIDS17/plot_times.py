import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1)


for alg, symb in zip(['DT','EFC', 'MLP', 'SVC'], ['o','d','^','h'] ):
    testing_time = []
    training_time = []
    for size in [1,2,3,4]:
        time = list(np.load("TimeData/Results/Size{}/{}_times.npy".format(size, alg)))
        training_time.append(time[0])
        testing_time.append(time[1])

    print(training_time)
    print(testing_time)
    if alg == "SVC":
        ax.plot([47107, 94214, 141321, 188428], training_time, label="SVM", marker=symb, linewidth=1, linestyle='-')
    
    else:
        ax.plot([47107, 94214, 141321, 188428], training_time, label=alg, marker=symb, linewidth=1, linestyle='-')

ax.set_title("Training times", fontsize=17)
ax.set_ylabel("Time (s)", fontsize=15)
ax.set_xticks([40000, 90000, 140000, 190000])
ax.set_xticklabels(['40000', '90000', '140000', '190000'],rotation=30, ha='right', rotation_mode='anchor')
ax.set_xlabel("Number of instances", fontsize=15)
ax.legend(loc=2)


fig.savefig("TimeData/Results/complexity.pdf", format='pdf', bbox_inches = "tight")
