import numpy as np
import pandas as pd
import os
import sys

os.system("python3 setup.py build_ext --inplace")
os.system("python3 Multi-class\ EFC/KDDCup99/train_test.py")
os.system("python3 Multi-class\ EFC/KDDCup99/colect_results.py")
