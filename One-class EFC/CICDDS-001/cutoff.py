import numpy as np
import pandas as pd
from classification_functions import train_energies, create_model
from matplotlib import pyplot as plt
import os


def define_cutoff(train_sample, Q, LAMBDA):
    train_data = np.load(train_sample, allow_pickle=True)
    model_normal, h_i = create_model(np.array(train_data,dtype=int), Q, LAMBDA)
    energies = train_energies(train_data, model_normal, h_i, Q)
    bins = np.histogram(np.hstack((energies)), bins=100)[1]
    energies.sort()
    cutoff = energies[int(len(energies)*0.95)]
    return cutoff
