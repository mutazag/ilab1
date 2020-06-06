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
con_bilayers = pd.read_csv(
    c.con_bilayers,
    header=None,
    names=['bilayer'])

# %% [markdown]
# ## get example of bilayers with T in the name

# %%
t_filter = con_bilayers.bilayer.str.contains('-T')
con_bilayers_T = con_bilayers[t_filter]

# %%
print(con_bilayers_T.apply(lambda x: x.str.contains('-T-')).sum())
print(con_bilayers_T.apply(lambda x: x.str.contains('-T')).sum())
print(con_bilayers_T.apply(lambda x: x.str.endswith('-T')))
print(con_bilayers_T.count())
# %%
# replace  -T- with _T_
# filter_T_ = con_bilayers.bilayer.str.contains('-T-')
# # con_bilayers_T['bilayer'].apply(lambda x: x.contains('-T-'))
# print(f'{filter_T_.sum()} bilayer with -T- in the middle')
# con_bilayers[filter_T_].loc[:,'bilayer'] = con_bilayers[filter_T_].bilayer.str.replace('-T-', '-T_')

# %%
# find ones with -T at the end
# filter_T = con_bilayers.bilayer.str.endswith('-T')
# print(f'{filter_T.sum()} bilayer end with -T')

# %%
con_bilayers['bilayer2'] = con_bilayers.bilayer.str.replace('-T-', '-T_')
# %%

# %% [markdown]
# ## logic for fixing the bilayer name
# after some experimentation, the following table represents the different cases to be addresses 
# and corresponding str replacement action
#
# | -T- in the middle | -T in the end | action |
# |:--- |:---: |---:|
# | Yes | No | replace('-T-', '-T_') |
# | Yes | Yes | replace('-T-', '-T_') |
# | No | No | replace ('-','_') |
# | No | Yes | replace('-','_',n=1) |
#
# to simplify the cases, When -T does not appear in the middle or the end
# then replace with n=1 (count = 1) so that we have the same code
# as for when -T also appears at the end
# %%
con_bilayers['bilayer3'] = con_bilayers.bilayer.apply(
    lambda x: x.replace('-T-', '-T_')
    if x.find('-T-') != -1
    else x.replace('-', '_', 1))

# %% spot checks
filter_middle_T = con_bilayers.bilayer3.str.contains('-T_')
filter_end_T = con_bilayers.bilayer3.str.endswith('-T')
# %%
con_bilayers[filter_middle_T].head()
# %%
con_bilayers[filter_end_T].head()
# %%
con_bilayers[filter_middle_T & filter_end_T].head()
# %%
# split monolayer names
con_bilayers[['monolayer1', 'monolayer2']
             ] = con_bilayers.bilayer3.str.split('_', expand=True)

#%%
#TODO:
# in this file, monolayers that end with -T dont have number of atoms
# in other files it would be something like -T1 
# so here monoalyer names will be fixed to have -T1 where needed before 
# joining them into a uid in the next step
con_bilayers['monolayer1'] = con_bilayers.monolayer1 \
    .apply(
    lambda x: x.replace('-T', '-T1') 
    if (x.endswith('-T')) 
    else x)


con_bilayers['monolayer2'] = con_bilayers.monolayer2 \
    .apply(
    lambda x: x.replace('-T', '-T1') 
    if (x.endswith('-T')) 
    else x)

con_bilayers[filter_middle_T & filter_end_T].head()
# %%
# calculate uid
con_bilayers['uid'] = con_bilayers[['bilayer3',
                                    'monolayer1', 'monolayer2']].apply(uid, axis=1)


#%%
# check after adding uid
con_bilayers[filter_middle_T & filter_end_T].head()
# %%
# select uid, monotlayer1 and 2 and save to file
con_bilayers[['uid', 'monolayer1', 'monolayer2']].to_csv(
    c.get_descriptorspath('con_bilayers_uid.csv'),
    index=False)

# %%
