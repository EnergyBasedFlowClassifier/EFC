os.makedirs("5-fold_sets/", exist_ok=True)

os.makedirs("5-fold_sets/Discretized", exist_ok=True)
os.makedirs("5-fold_sets/Non_discretized", exist_ok=True)
os.makedirs("5-fold_sets/Normalized", exist_ok=True)
os.makedirs("5-fold_sets/Results", exist_ok=True)

for sets in range(1,6):
    os.makedirs("5-fold_sets/Discretized/Sets{}".format(sets), exist_ok=True)
    os.makedirs("5-fold_sets/Non_discretized/Sets{}".format(sets), exist_ok=True)
    os.makedirs("5-fold_sets/Results/Sets{}".format(sets), exist_ok=True)

for removed in range(1,5):
    os.makedirs("5-fold_sets/Results_removing{}".format(removed), exist_ok=True)
    for sets in range(1,6):
        os.makedirs("5-fold_sets/Results_removing{}/Sets{}".format(removed, sets), exist_ok=True)
