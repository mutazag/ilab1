#%% [markdown]
# in this file, will explore with one compound and try to find it in both files 

#%%

import pandas as pd
import numpy as np
from utils.config import Config

c = Config() 
print(
    f"{c.predicted_300K}\n"
    f"{c.small_300K}\n"
    f"{c.predicted_18M}\n"
    f"{c.small_18M}\n")

#%%
v = c.validate_files() 
v

#%%
v.query('type == "small"')
v.query('origin == "300K"')

#%%
df = pd.read_csv(c.predicted_300K)

#%%
size_before = df.memory_usage()

#%% 
describe_before = df.describe()  
# notice missing values in Label column 

#%% 

# check for missing values 
df_missing = df.isna()
df_missing_counts = df_missing.sum()

print(df_missing_counts[df_missing_counts > 0])
#%% 
df.isna().mean().round(4) * 100


#%% 
# fill missing values for Monolayer and Label 
df.Label = df.Label.fillna(0)
df.Monolayer = df.Monolayer.fillna('')
df.isna().mean().round(4) * 100


#%% 
df = df.astype({
    "bilayer_index": np.int16, 
    "X1": np.int16, 
    "Y1": np.int16, 
    "Label": np.int16})

#%%


#%%
# df = df.astype({
#     "IE (J/m^2)":"float32", 
#     "Error": "float32", 
#     "Rel Err": "float32",
#     "C33 (GPa)": "float32",
#     "Error.1": "float32",
#     "Rel Err.1": "float32"
# })

#%%
size_after = df.memory_usage()
describe_after = df.describe()

print(f"{size_before.sum():,}")
print(f"{size_after.sum():,}")


#%% [markdown]
# ## print results after cleansing 

#%% 
# describe before and after 
print(describe_before)
print(describe_after)
#%%
# explore the memory usage of string columns 
df[["bilayer", "Monolayer 1", "Monolayer 2", "Monolayer"]].nunique()


#%% [markdown]

# explore the memory usage of string columns ...
#
# > bilayer        296835
#
# > Monolayer 1       770
#
# > Monolayer 2       770
#
# > Monolayer         771
#
#
# Monolayer 1, Monolayer 2 and Monolayer column could be coverted 
# to categorical variables 
# also worth exploring how to reduce the string size for bilayer columns

#%%
df['bilayer_len'] = df.bilayer.str.len()
df.bilayer_len.describe()

#max length of bilayer string is 31
#%%
