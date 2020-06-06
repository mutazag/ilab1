# %%
from utils.config import Config
from utils.data import Data
import pandas as pd

# %%
c = Config()
d = Data()


# %%
colnames = d.getDescriptorsColumnNames()
filename = '1l_atomicPLMF_6138structures.csv'

# %%
desc_6k_df = pd.read_csv(c.get_datapath(filename), index_col='Monolayer')

# %%
desc_6k_df_filtered = desc_6k_df[colnames]

# %%
desc_6k_df_filtered.to_csv(c.get_descriptorspath('descriptors_master_6k.csv'))

# %%
