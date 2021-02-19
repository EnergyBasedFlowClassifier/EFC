import os

os.mkdir("Data/")
os.mkdir("External_test/")

os.mkdir("Data/Discretized")
os.mkdir("Data/Non_discretized")
os.mkdir("Data/Results")
os.mkdir("External_test/Discretized")
os.mkdir("External_test/Non_discretized")

for exp in range(1,11):
    os.mkdir("Data/Discretized/Exp{}/".format(exp))
    os.mkdir("Data/Non_discretized/Exp{}/".format(exp))
    os.mkdir("Data/Results/Exp{}/".format(exp))
    os.mkdir("External_test/Discretized/Exp{}/".format(exp))
    os.mkdir("External_test/Non_discretized/Exp{}/".format(exp))
