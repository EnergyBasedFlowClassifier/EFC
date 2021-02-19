import os

os.mkdir("5-fold_sets/")

os.mkdir("5-fold_sets/Discretized")
os.mkdir("5-fold_sets/Non_discretized")
os.mkdir("5-fold_sets/Results")

for sets in range(1,6):
    os.mkdir("5-fold_sets/Discretized/Sets{}".format(sets))
    os.mkdir("5-fold_sets/Non_discretized/Sets{}".format(sets))
    os.mkdir("5-fold_sets/Results/Sets{}".format(sets))

for removed in range(1,13):
    os.mkdir("5-fold_sets/Results_removing{}".format(removed))
    for sets in range(1,6):
        os.mkdir("5-fold_sets/Results_removing{}/Sets{}".format(removed, sets))
