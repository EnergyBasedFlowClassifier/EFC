# EFC

Energy Flow Classifier (EFC) is a new method for flow-based network intrusion detection using the inverse Potts model.

This repository contains the implementation of EFC in a one-class version. The file **dca_functions.pyx** contains
functions to be used in model inferrence and the file **classification_functions.pyx** is where training and testing functions are
defined. Both files were written in Cython language and therefore must be built using **setup.py**. To do this, run the following command:

`python3 setup.py build_ext --inplace`

Lastly, **test_data.py** is an usage example of the classifier. Please note that it is
a fake script used only to exemplify the one-class EFC function calls. It will not run without
real data being loaded into the variables.

Observations:
* EFC requires discretized data
* The one-class EFC is trained with only normal samples.


# Experiments

The folder **One_class EFC** within the repository contains scripts used to
perform experiments on CICDDS-001, CICIDS2017 and CICDDoS2019 data sets. To reproduce the experiments,
please read the README.md file inside that folder.
The experiments results can be seen in
[A new method for flow-based network intrusion
detection using the inverse Potts model](https://arxiv.org/pdf/1910.07266.pdf)
