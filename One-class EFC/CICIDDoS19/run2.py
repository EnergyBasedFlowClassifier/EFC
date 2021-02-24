import os

os.system('cp ../CICIDS17/Dict_CICIDS17.npy ./')
exec(open("discretize.py").read())
exec(open("drop_duplicates.py").read())
exec(open("create_dir.py").read())
exec(open("build_train_test.py").read())
exec(open("build_external_set.py").read())

#cython build
os.system('python3 setup.py build_ext --inplace')

#experiment
exec(open("ML_algorithms.py").read())

#results
exec(open("colect_results.py").read())
exec(open("plot_energies.py").read())
