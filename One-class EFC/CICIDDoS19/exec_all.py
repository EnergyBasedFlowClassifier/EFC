import os

#pre process and sampling
exec(open("reduce.py").read())
exec(open("pre_process.py").read())
exec(open("Join_pre_processed.py").read())
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
