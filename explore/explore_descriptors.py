# %% 

from utils.config import Config
from utils.data import Data
import pandas as pd

# %%
c = Config()

#%%
desc_ie = pd.read_csv(c.descriptors_IE)
desc_c33 = pd.read_csv(c.descriptors_C33)

#%%
df_300K = Data().get300K()
#%%
desc_ie.shape
desc_c33.shape

#%% [markdown]
# # plan
# 1. check the unique monolayer ids in descriptors and compare with the 300K and 18M, how much is missing? 
# 2. some comparison is needed between the two descriptors files, do we take everything? 
# 3. now becuase i have the IE and C33, should be able to do some feature important analysis. 
# 4. sample the 300K file, is there a test for samlping 
# 5. fit a UMAP, fit different UMAPs
# 6. visualise 


# %% [markdown]
# # EDA 

# %% 

import seaborn as sns
import matplotlib.pyplot as plt 


#%%
sns.pairplot(desc_ie.iloc[:,:20])

#%%
g=sns.pairplot(desc_ie.iloc[:,:10],diag_kind="kde")
plt.show()
#%%

# %% [markdown]