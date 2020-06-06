#%% [markdown]
# process

#%%
from utils.data import Data
from utils.config import Config
import pandas as pd
import numpy as np

#%%
c = Config() 
d = Data()

#%% 
# load full feature df
df = d.get300K_features(indexed=False)
#%%
# we only need uid's for now
df_uids = df.uid
#%% 
# load con_bilayers_uid.csv
df_con = pd.read_csv(c.get_descriptorspath('con_bilayers_uid.csv'))
df_ie_dft = pd.read_csv(c.dft_IE_uid)
df_c33_dft = pd.read_csv(c.dft_C33_uid)

#%%
df_flags = df_uids
df_flags = pd.merge(
    df_flags, df_con,
    left_on='uid', 
    right_on='uid', 
    how='left', 
    suffixes=['uids', 'con'], 
    indicator='source_con')


df_flags = pd.merge(
    df_flags, df_ie_dft['uid'],
    left_on='uid', 
    right_on='uid', 
    how='left', 
    suffixes=['uids', 'dft_ie'], 
    indicator='source_dft_ie')


df_flags = pd.merge(
    df_flags, df_c33_dft['uid'],
    left_on='uid', 
    right_on='uid', 
    how='left', 
    suffixes=['uids', 'dft_c33'], 
    indicator='source_dft_c33')


#%%
both_count = df_flags.source_con.value_counts()['both']
con_count = df_con.shape[0]
print(
    f'{both_count:,d} commensurate predicted bilayers in 300K\n'
    f'from {con_count:,d} possible commensurate bilayers calcuated using DFT\n'
    f'difference is {con_count - both_count:,d}')
#%%
df_ie_count = df_flags.source_dft_ie.value_counts()['both']
df_c33_count = df_flags.source_dft_c33.value_counts()['both']
#%%
#create flag 
df_flags['commensurate'] = df_flags.source_con == 'both'
df_flags['dft_ie'] = df_flags.source_dft_ie == 'both'
df_flags['dft_c33'] = df_flags.source_dft_c33 == 'both'


#%%
# drop unwanted columns or keep ones we want 
#drop method 
df_flags.drop(columns=['monolayer1', 'monolayer2', 'source_con','source_dft_ie','source_dft_c33'], axis=1)  # could use inplace=True here 
#%%
#keep columsn we want 
df_flags[['uid', 'commensurate','dft_ie', 'dft_c33']]

#%%
# at this stage, i could just add the commensurate column to the original data set, i know it would work since the orger of the rows was not changed. but i prefer t use a merge method, this way i am 100% sure that i am matching the correct records
df_final = pd.concat(
    [
        df, 
        df_flags[['uid', 'commensurate', 'dft_ie', 'dft_c33']].rename(columns={'uid': 'uid_flags'})
    ], 
    axis=1,  # concat columns  
    sort=False)

#%%
# validating that concat was correct
(df_final.uid != df_final.uid_flags).sum()  # should be 0

#%%
# save to file

df_final.drop(columns=['uid_flags']).to_csv(
    c.get_datapath('300K_full_features_flags.csv'), 
    index=False)


# %%
