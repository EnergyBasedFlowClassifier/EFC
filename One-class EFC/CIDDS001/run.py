import os

# pre process and sampling
exec(open("reduce.py").read())
os.system('python3 preprocess_data.py all normal')
os.system('python3 preprocess_data.py all attacker')
os.system('python3 preprocess_data_external.py all suspicious')
os.system('python3 preprocess_data_external.py all unknown')
for i in range(1,11):
    os.system('python3 build_test_bank.py os {}'.format(i))
    os.system('python3 build_test_bank.py ext {}'.format(i))

exec(open("create_dir.py").read())
exec(open("ajust_sets.py").read())

#cython build
os.system('python3 setup.py build_ext --inplace')

#experiment
exec(open("ML_algorithms.py").read())

#results
exec(open("colect_results.py").read())
exec(open("plot_energies.py").read())

#time complexity experiment
exec(open("generateLargeFIles.py").read())
exec(open("ML_algorithms_times.py").read())

#results
exec(open("plot_times.py").read())
