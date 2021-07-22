# Experiments

To reproduce the experiments, first download ([CICIDS17](https://www.unb.ca/cic/datasets/ids-2017.html),
[CICDDoS19](https://www.unb.ca/cic/datasets/ddos-2019.html),
[CIDDS-001](https://www.hs-coburg.de/forschung/forschungsprojekte-oeffentlich/informationstechnologie/cidds-coburg-intrusion-detection-data-sets.html)) and extract the datasets in its corresponding directory using the unzip command. It is important to use the unzip command, as different forms of extraction can change the names of the directories used in the scripts.

In CIDDS-001, run **run.py** to execute the scripts for preprocessing the dataset, separating train/test sets, conducting training/testing and collecting results.
For CICDDoS19 and CICIDS17, the experiments must be executed by hand in parallel, since in the cross-domain experiment we use exchanged data.
To do so, run **run1.py** in both directories, then **run2.py** in both directories and finally **run3.py** in both directories.

The experiments with the last two datasets were performed on a machine with 32GB of RAM, therefore we recommend the usage of this setup to reproduce them.

After execution, the results will be in the Data/Results directory.
