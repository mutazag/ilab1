#%% 

import pandas as pd
import numpy as np
from utils.config import Config
from utils.layercodes import uid

c = Config() 
#%%
print(f"Processing file {c.predicted_18M}")

#%% 
#np.uint16	uint16_t	Unsigned integer (0 to 65535)
#np.uint32	uint32_t	Unsigned integer (0 to 4294967295) 
# np.uint32 range is large enough to hold the 18 M unique indecies for bilayers
# and data type of np.uint32 is half the size of default int64 used by pandas 
col_types = {
    "Unnamed: 0": np.uint16, 
    "Unnamed: 0.1": np.uint16,
    "index_bilayer": np.uint32,
    "FlagX1": np.uint16, 
    "FlagY1": np.uint16, 
}
# specifying uint data types reduced the size in memory by 20%
#%% 
df = pd.read_csv(c.predicted_18M, dtype=col_types, nrows=1000000)
print(df.info())
#%%
# apply uids 

df['uid'] = df[['bilayer','monolayer1','monolayer2']].apply(uid, axis=1)

#%%
