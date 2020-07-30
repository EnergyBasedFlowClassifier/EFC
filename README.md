# EFC

Energy Flow Classifier (EFC) is a new method for flow-based network intrusion detection using the inverse Potts model.

This repository contains files that implements EFC in a one-class version and a multi-class version. These files are *dca_functions*, *classification_functions* and *test_data*. The first one contains
functions to be used in the model inferrence for both versions. The second one is where training and testing function are defined for each classifier. The last file is an example
of usage of the classifiers.

Observations:
* EFC requires discretized data
* The one-class EFC is trained with only normal samples.


# Experiments

The folders within the repository contains scripts used to
perform experiments with EFC on CICDDS-001, CICIDS2017 and CICDDoS2019 datasets.
The experiments conducted for the one-class EFC are in the One-class folder and their results can be seen in the paper
[A new method for flow-based network intrusion
detection using the inverse Potts model](https://arxiv.org/pdf/1910.07266.pdf)

The other folder, named *Hybrid EFC*, contains the scripts for experiments with an hybrid system composed of the one-class EFC and the multi-class EFC.



