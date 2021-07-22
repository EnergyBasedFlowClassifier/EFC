import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from statistics import mean, stdev
from math import sqrt

plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1,2, figsize=(12.8, 4.8))


for alg, symb in zip(['RF','GaussianNB','KNN',  'Adaboost', 'DT','EFC'], ['o','d','^','h','s','>'] ):
    testing_time_mean = []
    testing_time_error = []
    training_time_mean = []
    training_time_error = []
    for size in [0,1,2,3]:
        testing_time = []
        training_time = []
        for exp in range(1,11):
            time = list(np.load("TimeData/Results/Size{}/Exp{}/{}_times.npy".format(size, exp, alg)))
            training_time.append(time[0])
            testing_time.append(time[1])
        training_time_mean.append(mean(training_time))
        training_time_error.append((stdev(training_time)/sqrt(len(training_time)))*1.96)
        testing_time_mean.append(mean(testing_time))
        testing_time_error.append((stdev(testing_time)/sqrt(len(testing_time)))*1.96)


    ax[0].errorbar([16000, 32000, 48000, 64000], training_time_mean, yerr = training_time_error, label=alg, marker=symb,
             markersize=5, linewidth=1, linestyle='-')
    ax[1].errorbar([4000, 8000, 12000, 16000], testing_time_mean, yerr = testing_time_error, label=alg, marker=symb,
             markersize=5, linewidth=1, linestyle='-')

ax[0].set_title("Training times", fontsize=17)
ax[0].set_ylim([-0.1, 3])
ax[0].set_ylabel("Time (s)", fontsize=15)
ax[0].set_xlabel("Number of samples", fontsize=15)
ax[0].legend(loc=2)


ax[1].set_title("Testing times", fontsize=17)
ax[1].set_ylim([-0.01, 0.7])
ax[1].set_ylabel("Time (s)", fontsize=15)
ax[1].set_xlabel("Number of samples", fontsize=15)
ax[1].legend(loc=2)
fig.savefig("TimeData/Results/complexity.pdf", format='pdf', bbox_inches = "tight")
