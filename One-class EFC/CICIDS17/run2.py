import os

# discretize for external test
os.system('cp ../CICIDDoS19/Dict_CICDDoS19.npy ./')
exec(open("discretize.py").read())
exec(open("drop_duplicates.py").read())
exec(open("create_dir.py").read())

# sampling
exec(open("build_train_test.py").read())
exec(open("build_external_set.py").read())
