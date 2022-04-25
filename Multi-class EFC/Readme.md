# Experiments

To reproduce the experiments, first download and extract [CICIDS17](https://www.unb.ca/cic/datasets/ids-2017.html) in its corresponding directory using the unzip command. You will also need to download and extract the [version](https://github.com/zhangzhao156/scalable-NIDS) of CICIDS2017 made available by Zhang et al. It is important to use the unzip command, as different forms of extraction can change the names of the directories used in the scripts. Then, you need to build EFC Cython module in the root directory, using the command
`python3 setup.py build_ext --inplace`

To preprocess, discretize, sample and execute the experiments in CICIDS2017 folder, run the **exec.py** script.
**ML_algorithms.py** is the script in which training and testing of all classical ML algorithms and EFC happen, in 5-fold cross-validation. **ML_algorithms_unknown.py**
performs training and testing of the algorithms removing one type of attack from training (one at a time), also in 5-fold cross-validation.

To run the experiment in the Literature folder, where we compare EFC with other open-set methods like the one by Zhang et al, run the **exec.py** script.



