# EFC

The Energy-based Flow Classifier (EFC) is a new classification method to be used in network intrusion detection systems.

This repository contains two EFC implementations: a single-class version and a multi-class version. To use the algorithm in either version, you need to download the files **dca_functions.pyx**, **classification_funtions_seq.pyx**, **classification_functions_parallel.pyx** and **setup.py**. 

- **dca_functions.pyx** - contains auxiliary functions used by EFC

- **classification_funtions_seq.pyx** - contains EFC's training and testing functions in sequential form

- **classification_funtions_parallel.pyx** - contains EFC's training and testing functions using parallelism

- **setup.py** - contains building instructions to the Cython modules

Since EFC is implemented in Cython language, it needs to be built with the following command:

`python3 setup.py build_ext --inplace`

After building, one can use EFC as shown in **usage_example.py**.

Observations:
* EFC requires discretized data as input
* The one-class EFC is trained with only benign samples (class 0).
* To change between sequential or parallel versions of EFC edit **setup.py** according to the comments on the file.
* To use the scipts from this repository, the following dependencies are required: Numpy, Scipy, Cython, Pandas, Scikit-learn and Seaborn.

**Note:** This repository contains scripts from the first studies of the EFC method. Nowdays, EFC is available as a scikit-learn compatible [package](https://github.com/EnergyBasedFlowClassifier/EFC-package).

# Experiments

The folder **One_class EFC** within the repository contains scripts used to
perform experiments with the Single-class EFC with CICDDS-001, CICIDS2017 and CICDDoS2019 data sets. To reproduce this experiments,
please read the README.md file inside that folder.
The experiments results can be seen in
[A new method for flow-based network intrusion
detection using the inverse Potts model](https://ieeexplore.ieee.org/document/9415676)

The folder **Multi_class EFC** contains scripts used to
perform experiments with the Multi-class EFC with CICDDS-001 and CICIDS2017 datasets. To reproduce this experiments,
please read the README.md file inside that folder.

