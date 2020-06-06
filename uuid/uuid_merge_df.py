# %%
import pandas as pd
from pandas.api.types import union_categoricals
import numpy as np
from utils.config import Config
import matplotlib.pyplot as plt

#%%
c = Config()
c.validate_files()

#%% 
#prepare new columns,labels
df1_columns_index = [2,3,5,7,8,9,10,11,12,15]
df1_columns_labels = ['bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err','uid (index)']

df2_columns_index = [4,5,6,7,8,11,9,10,12,16]
df2_columns_labels = ['bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err','uid (index)']

# %%
df1 = pd.read_csv(c.uid_300K) #, nrows=10000)
#%%
df1.index = df1.uid
df1 = df1.iloc[:, df1_columns_index[:-1]]
df1.columns = df1_columns_labels[:-1]

# %%
df2 = pd.read_csv(c.uid_18M)# , nrows=100000)
# %%
df2.index = df2.uid
df2 = df2.iloc[:,df2_columns_index[:-1]]
df2.columns = df2_columns_labels[:-1]

# %%
print(f"size of df1: {df1.shape[0]:,}")

# %%
print(f"size of df2: {df2.shape[0]:,}")

#%% 
# merge cats1 and cats2 series 
df_combine = pd.merge( 
    left=df2, 
    right=df1, 
    how='outer',
    on='uid', 
    suffixes=('_18M', '_300K'), 
    indicator='source'
)

#%%
#explore the common ones 
filter_both = (df_combine.source == 'both')

# look for cases where bilayer names were the otherway around in the datasets
filter_both = filter_both & (df_combine.bilayer_18M != df_combine.bilayer_300K)
df_filtered = df_combine[filter_both][[
    'bilayer_18M','monolayer1_18M', 'monolayer2_18M', 
    'bilayer_300K','monolayer1_300K','monolayer2_300K', 
    'source']]

print(df_filtered.shape)
df_filtered
#%%
# cats.groupby('source').nunique()
# cats.source.value_counts()|
df_combine_counts = df_combine.groupby('source').size()
df_combine_counts

#%%
df_combine_counts.plot.bar()
plt.yscale('log')

#%%
print(f"size of df1: {df1.shape[0]:,}")
print(f"size of df2: {df2.shape[0]:,}")


#%%
df_combine_counts[~df_combine_counts.index.str.contains('left')].plot.bar()
plt.yscale('log')
#%%
# save combined df 
df_combine.to_csv("data/ML_IE_C33/df_combined.csv")

#%%
