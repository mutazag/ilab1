# %%
import pandas as pd
from pandas.api.types import union_categoricals
import numpy as np
from utils.config import Config

#%%
c = Config()
c.validate_files()

# %%
df = pd.read_csv(c.predicted_300K)

# %%
df = df[["Monolayer 1", "Monolayer 2"]]

# %%
cats1 = pd.unique(df.to_numpy().ravel())
print(f"size of cats1: {cats1.size}")
# %%
df2 = pd.read_csv(c.predicted_18M)
# %%
df2 = df2[['monolayer1', 'monolayer2']]

# %%
cats2 = pd.unique(df2.to_numpy().ravel())
print(f"size of cats2: {cats2.size:,}")

#%% 
# merge cats1 and cats2 series 

cats = pd.merge(
    pd.Series(cats2, name='monolayer'), 
    pd.Series(cats1, name='monolayer'), 
    how='outer', 
    indicator='source')

#%%
# cats.groupby('source').nunique()
# cats.source.value_counts()
cats.groupby('source').count()

#%%
cats.groupby('source')['monolayer'].count().plot.bar()

#%%
print(f"size of cats2: {cats2.size:,}")
print(f"size of cats1: {cats1.size}")

#%%
