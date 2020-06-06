# %%[markdown]
# # Prepare a fully featured 18M file

# join the 18M with master descriptors to produce a fully 
# fetaured 18M file

#%%
import sys
import os
import pathlib
utils_path = pathlib.Path(os.getcwd() + '/utils')  # i suspect this one is not needed
print(utils_path.exists())

sys.path.append(str(utils_path))  # may not be necessary
sys.path.append(os.getcwd())  # i thnk this is the one that works 
print(sys.path)
#TODO: test which of the above two sys.path.append was actually the correct one 

# %%
from utils.data import Data
from utils.config import Config
import pandas as pd
import numpy as np
import tqdm
# %%
d = Data()
feature_columns = d.getDescriptorsColumnNames()
# %%
# load desc_master, reset index and rename monolayer to lowercase
desc_master = d.getDescriptorsMaster_6k(indexed=False)

# %%
# load df 18M, reset df index to be used for melt operation
df = d.get18M(indexed=False)
# df.reset_index(level=0, inplace=True) # no nee fo this step
columns_to_bring_back = ['uid', 'bilayer', 'monolayer1', 'monolayer2',
                         'IE', 'IE_error', 'IE_rel_error', 'C33', 'C33_error', 'C33_rel_err']

# %%
# melt df, will duplicate uids
df_long = pd.melt(
    df,
    id_vars=['uid'],
    value_vars=['monolayer1', 'monolayer2'],
    var_name='layer',
    value_name='layer_name').sort_values('uid').drop(['layer'], axis=1)

# %% [markdown]
# # sampling section
# %%
# quick look at relevant descriptors for sample
desc_master.head(4)

#%%
df_length = df_long.shape[0] 
chunk_size = 500000
i = 0
df_features_list = []

for i in range(0, df_length, chunk_size): 
    chunk_end = i + chunk_size
    print(i, chunk_end)
    df_temp = df_long[i:chunk_end]
    df_f_temp = df_temp.rename(columns={'layer_name': 'monolayer'}) \
        .join(desc_master.set_index('monolayer'), on='monolayer') \
        .groupby('uid').sum()
    df_features_list.append(df_f_temp)

df_features = pd.concat(df_features_list)
# %%
# join operaetion
# must ensure that indexes match for the two data sets, otherwise must use merge operaetion


# df_features = df_long[:100].rename(columns={'layer_name': 'monolayer'}) \
#     .join(desc_master.set_index('monolayer'), on='monolayer') \
#     .groupby('uid').sum()


#%%

df_length = df.shape[0] 
chunk_size = 500000
i = 0
df_full_list = []

for i in range(0, df_length, chunk_size): 
    chunk_end = i + chunk_size
    print(i, chunk_end)
    df_temp = df[i:chunk_end]
    df_f_temp = df_temp.join(df_features, on='uid')
    df_full_list.append(df_f_temp)

df_full = pd.concat(df_full_list)
# df_full = df.join(df_features, on='uid')


#%% 
df_full.to_csv(Config().get_datapath('18M_full_features.csv'), index=False)

#%% [markdown]
# ## cross check outcomes 
#%% 
# get a random sample of uids 
uids = df.uid.sample(n=5)
print(f'randome set of uids:  {", ".join(uids)}')
print('\n\n\nfeatures for selected uids')
sample_features = df_features[df_features.index.isin(uids)].iloc[:, :3].sort_index()
print(sample_features)
sample_df_full = df_full[df_full.uid.isin(uids)].iloc[:, :12].sort_values('uid')
print('\n\n\nfull feature set')
print(sample_df_full)
#%%
