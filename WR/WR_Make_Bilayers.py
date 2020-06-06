# %%
from pathlib import Path
import sys
import os
from sklearn.preprocessing import binarize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn import linear_model
from sklearn import model_selection, preprocessing
import seaborn as sns
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
from scipy import interp
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn import svm, datasets
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np

# %%
print(__doc__)

# %%
# import xgboost as xgb
color = sns.color_palette()

Number_Monolayers = 6138

# %%

data_path = Path('./data/WR/')

# %%
# Create directory
# Create target Directory if don't exist
if not os.path.exists(data_path):
    os.mkdir(data_path)
    os.mkdir(data_path / 'DATA_SETS/')
    os.mkdir(data_path / 'Figs/')
    os.mkdir(data_path / 'SavedModels/')
    os.mkdir(data_path / 'LASSO_Converged/')
    print("Directory ", data_path, " Created ")
else:
    print("Directory ", data_path, " already exists")


# %matplotlib inline

# %%
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', 3000)

# %%
filename = data_path / \
    ("1l_atomicPLMF_" + str(Number_Monolayers) + "structures.csv")

# %%
if filename.exists() is False:
    print(f"file does not exist: {filename} ")
    downloadurl = "https://www.dropbox.com/sh/b1xjnrwyvnxvufy/AACQd3nAgPV3ASVq4EFlrhwra/" \
        "LASSO_BR2_1?dl=0&preview=1l_atomicPLMF_6138structures.csv"
    print(f"use curl to download {downloadurl}")
else:
    print(f"file exists: {filename}")


# %%
pd.read_table(filename, nrows=1000, low_memory=False, header=None, sep='#')

# %%
skipcount = 100
with open(filename, mode='r') as f1:
    i = 0
    lines = []
    for line in f1:
        if (i % skipcount == 0):
            lines.append(line)
        i += 1


# %%
# read file with monolayers names and descriptors
monolayer_descriptors = pd.read_csv(filename, header=0)
titles = pd.read_csv(filename, header=None)
numMonolayerColumns = monolayer_descriptors.shape[1]
numMonolayerRecords = monolayer_descriptors.shape[0]

print('numMonolayerColumns', numMonolayerColumns)
print('numMonolayerRecords', numMonolayerRecords)

# %%
# read file with bilayers names and target values
BilayerProperty = pd.read_csv(data_path / "C33_DFT.csv", header=0)

numBilayerRecords = BilayerProperty.shape[0]
# print('numBilayerRecords',numBilayerRecords)

# %%
bilayers = BilayerProperty.iloc[:, 0]
# print('bilayers',bilayers)
monolayers = monolayer_descriptors.iloc[:, 0]
# print('monolayers',monolayers)


# %%
dataset = []
mislabeled = []


for b in bilayers:
    print(b)
    bt = b.split("_")
    b_d = BilayerProperty.loc[BilayerProperty.Bilayer == b]
    bilayer_record = []
    m1 = monolayer_descriptors.loc[monolayer_descriptors.Monolayer == bt[0]]
    m2 = monolayer_descriptors.loc[monolayer_descriptors.Monolayer == bt[1]]
    i = 1
    try:
        sum = m1.iloc[0, i] + m2.iloc[0, i]
        for i in range(1, numMonolayerColumns):
            sum = m1.iloc[0, i] + m2.iloc[0, i]
            bilayer_record += [sum]
        bilayer_record += [b_d.iloc[0, 1]]
        dataset += [bilayer_record]
    except:
        try:
            m1.iloc[0, 1]
            print("cannot find", bt[1], "in 1l_atomicPLMF")
            mislabeled += bt[1]

        except:
            print("cannot find", bt[0], "in 1l_atomicPLMF")
            mislabeled += bt[0]
#    for i in range(1,numMonolayerColumns):
#        sum = m1.iloc[0,i] + m2.iloc[0,i]
#        bilayer_record += [sum]
#    bilayer_record += [b_d.iloc[0,1]]
#    dataset += [bilayer_record]


# sys.exit(-1)

# df_dataset=pd.read_csv("PLMF.csv",header=0)
# df_dataset=pd.read_csv("PLMF.csv")

# %%
df_dataset = pd.DataFrame(dataset)
df_mislabeled = pd.DataFrame(mislabeled)

# %%
df_dataset.to_csv(data_path / 'mutaz' / "PLMF.csv", header=True)


df_mislabeled.to_csv(data_path / 'mutaz' / "PLMF_mislabeled.csv", header=True)
