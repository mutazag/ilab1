# %%
from utils.data import Data
import pandas as pd 
import numpy as np


# %% 

df = Data().get300K(indexed=True)
df.head(1)
# %%
monolayers = df.monolayer1.unique()
print(f"{monolayers.size} unique monolayers")

# %% 
dict ={
    'monolayer' :[],
    'corr': [], 
    'IE_mean' : [], 
    'C33_mean' : []
}
#%%
for monolayer_name in monolayers[:100]: 
    print(monolayer_name)
    monolayer_filter = (df.monolayer1 == monolayer_name) | (df.monolayer2 == monolayer_name)
    print(monolayer_filter.sum())
    df_filtered = df[monolayer_filter].loc[:,['IE','C33']]
    df_describe = df_filtered.describe()
    df_describe['stat'] = df_describe.index
    # df_describe.reset_index(level=0, inplace=True)
    df_corr = df_filtered.corr()
    dict['monolayer'].append(monolayer_name)
    dict['corr'].append(abs(df_corr.iloc[0,1]))
    dict['IE_mean'].append(df_describe.loc['mean'].IE)
    dict['C33_mean'].append(df_describe.loc['mean'].C33)

#%%
df_summary = pd.DataFrame(dict)#.set_index('monolayer')




#%%
df_summary.sort_values('IE_mean', ascending=False)

#%%
df_summary.IE_mean.hist()

#%%
df_summary.IE_mean.plot.density()

#%%
df_summary.C33_mean.plot.density()

#%%
df_summary.C33_mean.hist(bins=100)

#%%
df_summary[['IE_mean','C33_mean']].plot(x='IE_mean', y='C33_mean',kind='scatter')

#%%
