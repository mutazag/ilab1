# %%
import pandas as pd
from pandas.api.types import union_categoricals
import numpy as np
from utils.config import Config

#%%
c = Config()
c.validate_files()

# %%
df = pd.read_csv(c.uid_300K)
# %%
uid1 = df.uid
# %%
print(f"size of uid1: {uid1.size:,}")
# %%
df2 = pd.read_csv(c.uid_18M)
# %%
uid2 = df2.uid

# %%
print(f"size of uid2: {uid2.size:,}")

#%% 
# merge cats1 and cats2 series 

uid = pd.merge(
    uid2,
    uid1,
    how='outer', 
    indicator='source')

#%%
# cats.groupby('source').nunique()
# cats.source.value_counts()
uid.groupby('source').count()

#%%
uid.groupby('source')['uid'].count().plot.bar()

#%%
print(f"size of uid1: {uid1.size:,}")
print(f"size of uid2: {uid2.size:,}")

#%%
uid_count = uid.groupby('source').uid.count()
uid_count[~uid_count.index.str.contains('left')].plot.bar()
#%%
