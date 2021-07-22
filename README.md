# EFC

The Energy-based Flow Classifier (EFC) is a new classification method to be used in network intrusion detection systems.

This repository contains two EFC implementations: a single-class version and a multi-class version. To use the algorithm in either version, you need to download the files **dca_functions**, **classification_funtions_seq.pyx**, **classification_functions_parallel.pyx** and **setup.py**. 

- **dca_functions** - contains auxiliary functions used by EFC

- **classification_funtions_seq.pyx** - contains EFC's training and testing functions in sequential form

- **classification_funtions_parallel.pyx** - contains EFC's training and testing functions using parallelism

- **setup.py** - contains building instructions to the Cython modules

Since EFC is implemented in Cython language, it needs to be built with the following command:

`python3 setup.py build_ext --inplace`

After building, one can use EFC as shown in **usage_example.py**.


Observations:
* EFC requires discretized data
* The one-class EFC is trained with only benign samples (class 0).
* To change between sequential or parallel versions of EFC edit **setup.py** according to the comments on the file.



# Experiments

The folder **One_class EFC** within the repository contains scripts used to
perform experiments with the Single-class EFC with CICDDS-001, CICIDS2017 and CICDDoS2019 data sets. To reproduce this experiments,
please read the README.md file inside that folder.
The experiments results can be seen in
[A new method for flow-based network intrusion
detection using the inverse Potts model](https://arxiv.org/pdf/1910.07266.pdf)

The folder **Multi_class EFC** contains scripts used to
perform experiments with the Multi-class EFC with CICDDS-001, CICIDS2017 and KDDCup99 data sets. To reproduce this experiments,
please read the README.md file inside that folder.

