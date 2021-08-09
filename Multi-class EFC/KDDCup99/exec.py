import os

exec(open("pre_process.py").read())
exec(open("drop_unknown_duplicates.py").read())

os.system("python3 normalize.py 0")
os.system("python3 normalize.py 1")

os.system("python3 discretize.py 0")
os.system("python3 discretize.py 1")

os.system("python3 train-test.py 0")
os.system("python3 train-test.py 1")
