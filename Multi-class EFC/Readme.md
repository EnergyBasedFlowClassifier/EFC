# Experiments

To reproduce the experiments, first download and extract [CICIDS17](https://www.unb.ca/cic/datasets/ids-2017.html) and
[CIDDS-001](https://www.hs-coburg.de/forschung/forschungsprojekte-oeffentlich/informationstechnologie/cidds-coburg-intrusion-detection-data-sets.html) in its corresponding directories using the unzip command. It is important to use the unzip command, as different forms of extraction can change the names of the directories used in the scripts. Then, you need to build EFC Cython module in the root directory, using the command
`python3 setup.py build_ext --inplace`

## CICIDS17 and CIDDS001
To preprocess, discretize, sample and execute, run the **exec.py** script inside each folder.
**ML_algorithms.py** is the script in which training and testing of all classical ML algorithms and EFC happen, in 5-fold cross-validation. **ML_algorithms_unknown.py**
performs training and testing of EFC and RF removing one type of attack from training (one at a time), also in 5-fold cross-validation.



