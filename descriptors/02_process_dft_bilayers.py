# %%[markdown]
# # Process Con Bilayers

# split and generate proper bilayer uids
#%%
import sys
import os
import pathlib
utils_path = pathlib.Path(os.getcwd() + '/utils')
print(utils_path.exists())

sys.path.append(str(utils_path))
sys.path.append(os.getcwd())
print(sys.path)

# %%
from utils.data import Data
from utils.config import Config
import pandas as pd
import numpy as np
from utils.layercodes import uid


# %%
c = Config()
# %%
dft_c33 = 'C33_DFT.csv'
dft_ie = 'IE_DFT.csv'

dft_c33_df = pd.read_csv(
    c.get_datapath(dft_c33),
    header=0,
    names=['bilayer', 'C33_DFT'])   


dft_ie_df = pd.read_csv(
    c.get_datapath(dft_ie),
    header=0,
    names=['bilayer', 'IE_DFT'])       

# %%
# split monolayer names
dft_c33_df[['monolayer1', 'monolayer2']] = dft_c33_df.bilayer.str.split('_', expand=True)
dft_ie_df[['monolayer1', 'monolayer2']] = dft_ie_df.bilayer.str.split('_', expand=True)



#%%
# calculate uid
dft_c33_df['uid'] = dft_c33_df[['bilayer','monolayer1', 'monolayer2']].apply(uid, axis=1)
dft_ie_df['uid'] = dft_ie_df[['bilayer','monolayer1', 'monolayer2']].apply(uid, axis=1)


# %%
# select uid, monotlayer1 and 2 and save to file
dft_c33_df[['uid', 'monolayer1', 'monolayer2', 'C33_DFT']].to_csv(c.dft_C33,index=False)
dft_ie_df[['uid', 'monolayer1', 'monolayer2', 'IE_DFT']].to_csv(c.dft_IE,index=False)

# %%
