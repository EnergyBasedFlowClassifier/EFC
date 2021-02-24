import os

#pre process and discretize
os.mkdir("All_files_pre_processed/")
exec(open("clean.py").read())
exec(open("pre_process.py").read())
exec(open("Join_pre_processed.py").read())
exec(open("get_intervals.py").read())
