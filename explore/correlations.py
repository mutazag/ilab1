#%%
from utils.config import Config

import pandas as pd
import time

#%%

c = Config() 
c.validate_files()

#%%
df1 = pd.read_csv(c.uid_300K)
df1_columns_index = [15, 2,3,5,7,8,9,10,11,12]
df1_columns_labels = ['uid','bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err']
df1 = df1.iloc[:, df1_columns_index]
df1.columns = df1_columns_labels

#%%
corr_dict = {}
# monolayer = 'Hf3Te2'
monolayers = df1.monolayer1.unique()


#%% 
for monolayer_name in monolayers:
    timer = time.time()
    # 1 filter by monolayer name in both columns 
    monolayer_filter = (df1.monolayer1 == monolayer_name) | (df1.monolayer2 == monolayer_name)
    df_filtered =df1[monolayer_filter]

    # 2 calculate corr between IE and C33
    corr_monolayer = df_filtered[['IE','C33']].corr()
    # 3 add result to a dict with monolayer name as key 
    corr_dict[monolayer_name] = corr_monolayer.iloc[0,1]
    print(f"{monolayer_name}, corr {corr_dict[monolayer_name]},  timer: {time.time() - timer}")

#%%
# 4 repeat for all monolayers then create a df based on dict 
df_corr = pd.DataFrame.from_dict(corr_dict, orient='index', columns=['IE_C33_corr'])
df_corr.index.name = 'monolayer'

#%%
df_corr['abs_corr'] = df_corr.IE_C33_corr.abs()

#%%
df_corr.abs_corr.plot.density()

#%%
df1[['IE', 'C33']].plot.scatter(0,1)


#%%
