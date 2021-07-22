import os

#pre process and sampling
os.mkdir("Reduced/")
os.mkdir("Reduced/01-12/")
os.mkdir("Reduced/03-11/")
exec(open("reduce.py").read())
os.mkdir("Pre_processed/")
os.mkdir("Pre_processed/01-12/")
os.mkdir("Pre_processed/03-11/")
exec(open("pre_process.py").read())
exec(open("Join_pre_processed.py").read())
exec(open("get_intervals.py").read())
