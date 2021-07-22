#Experiments

To reproduce the experiments, first download ([CICIDS17](https://www.unb.ca/cic/datasets/ids-2017.html),
[CIDDS-001](https://www.hs-coburg.de/forschung/forschungsprojekte-oeffentlich/informationstechnologie/cidds-coburg-intrusion-detection-data-sets.html)) and extract the datasets in its corresponding directory using the unzip command. It is important to use the unzip command, as different forms of extraction can change the names of the directories used in the scripts.

##CICIDS17 and CIDDS001
To preprocess, discretize and sample the dataset, run the **exec.py** script inside each folder.
To run the experiments, **ML_algorithms.py** does the training and testing of all classical ML algorithms in 5-fold cross-validation; and **ML_algorithms_unknown.py**
performs training and testing by removing one type of attack from training (one at a time). This second experiment was conducted only for EFC and RF, also in 5-fold cross-validation.


