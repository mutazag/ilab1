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
import time
# %%
c = Config()
d = Data()
feature_columns = d.getDescriptorsColumnNames()
# %%
# load desc_master, reset index and rename monolayer to lowercase
desc_master = d.getDescriptorsMaster_6k(indexed=False)

# %%
# load df 18M, reset df index to be used for melt operation
# df = d.get18M(indexed=False)

chunk_size = 500000
for i, df_chunk in enumerate(pd.read_csv(c.uid_18M, chunksize=chunk_size)):
    # re-order and label columns
    df_chunk = df_chunk.iloc[:, d.columns_index_18M]
    df_chunk.columns = d.columns_labels_18M

    range_str = f'Begin processing: {i:03d} - {i*chunk_size} to '\
        f'{(i*chunk_size)+df_chunk.shape[0]-1}'

    print(range_str)

    timer = time.time()

    # melt chunk
    print(f'melt: {time.time() - timer}')
    df_long = pd.melt(
        df_chunk,
        id_vars=['uid'],
        value_vars=['monolayer1', 'monolayer2'],
        var_name='layer',
        value_name='layer_name').sort_values('uid').drop(['layer'], axis=1)

    # aggregate descriptors 
    print(f'calc bilayer descriptors: {time.time() - timer}')
    df_features = df_long.rename(columns={'layer_name': 'monolayer'}) \
        .join(desc_master.set_index('monolayer'), on='monolayer') \
        .groupby('uid').sum()

    # join chunk with df_features
    print(f'join predictions and features: {time.time() - timer}') 
    df_full = df_chunk.join(df_features, on='uid')    

    # save chunk to file 
    print(f'save chunck to file chunk_{i:03d}.csv: {time.time() - timer}')
    df_full.to_csv(c.get_datapath(f'18M_full_features/chunk_{i:03d}.csv'), index=False) 
    print(f"End timer: {time.time() - timer}")


#%% 
print('FINISHED')
