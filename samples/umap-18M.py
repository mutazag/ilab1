import sys 
import os
import pathlib
import time
print(sys.version)

utils_path = pathlib.Path(os.getcwd() + '/utils')  # i suspect this one is not needed
print(utils_path.exists())
print(os.getcwd())
#sys.path.append(str(utils_path))  # may not be necessary
#sys.path.append(os.getcwd())  # i thnk this is the one that works 
sys.path.append('../') # this one is one level up so we can see the utils lib
print(sys.path)

import numpy as np
import sklearn
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.data import Data
from utils.config import Config

import umap
import numba

d = Data()
# v2 = True: loading the dataset with the con flag
df = d.get18M_features(v2=True)

print('END')