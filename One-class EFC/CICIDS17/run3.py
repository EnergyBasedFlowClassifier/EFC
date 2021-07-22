import os

#cython build
os.system('python3 setup.py build_ext --inplace')

#experiment
exec(open("ML_algorithms.py").read())

#results
exec(open("colect_results.py").read())
exec(open("plot_energies.py").read())
