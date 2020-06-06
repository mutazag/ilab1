#%%
from utils.data import Data
import pandas as pd 

#%%
d = Data()

#%%
df = d.get300K()
#%%
print(df.head(1))
print(df.info())
print(df.shape)
#%%
df2 = d.get18M()
print('end')

#%%
