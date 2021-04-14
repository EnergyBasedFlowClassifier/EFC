import os

os.makedirs("Data/", exist_ok=True)
os.makedirs("External_test/", exist_ok=True)

os.makedirs("Data/Discretized", exist_ok=True)
os.makedirs("Data/Non_discretized", exist_ok=True)
os.makedirs("Data/Results", exist_ok=True)
os.makedirs("External_test/Discretized", exist_ok=True)
os.makedirs("External_test/Non_discretized", exist_ok=True)

for exp in range(1,11):
    os.makedirs("Data/Discretized/Exp{}/".format(exp), exist_ok=True)
    os.makedirs("Data/Non_discretized/Exp{}/".format(exp), exist_ok=True)
    os.makedirs("Data/Results/Exp{}/".format(exp), exist_ok=True)
    os.makedirs("External_test/Discretized/Exp{}/".format(exp), exist_ok=True)
    os.makedirs("External_test/Non_discretized/Exp{}/".format(exp), exist_ok=True)
