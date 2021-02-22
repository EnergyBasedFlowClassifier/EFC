import os

#pre process and save discretization intervals
exec(open("reduce.py").read())
exec(open("pre_process.py").read())
exec(open("Join_pre_processed.py").read())
exec(open("get_intervals.py").read())
