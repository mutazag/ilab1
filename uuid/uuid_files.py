#%%
import pandas as pd
import numpy as np
from utils.config import Config
from utils.layercodes import add_uid, add_uid2

#%%
c = Config()
c.validate_files()

#%%
print(f"Processing {c.predicted_300K}, output: {c.uid_300K}")
# add_uid(c.predicted_300K, ['bilayer', 'Monolayer 1', 'Monolayer 2'], c.uid_300K)
print(f"finished: {c.uid_300K}")

#%%
new_uid_filename = c.get_datapath("300K_uid_2.csv")
print(f"Processing method2 {c.predicted_300K}, output: {new_uid_filename}")
add_uid2(c.predicted_300K, ['bilayer', 'Monolayer 1', 'Monolayer 2'], new_uid_filename)
print(f"finished: {new_uid_filename}")


#%%
print(f"Processing {c.predicted_18M}, output: {c.uid_18M}")
# add_uid(c.predicted_18M, ['bilayer', 'monolayer1', 'monolayer2'], c.uid_18M)
print(f"finished: {c.uid_18M}")


#%%
new_18M_filename = c.get_datapath("18M_uid_2.csv")
print(f"Processing method2 {c.predicted_18M}, output: {new_18M_filename}")
add_uid2(c.predicted_18M, ['bilayer', 'monolayer1', 'monolayer2'], new_18M_filename)
print(f"finished: {new_18M_filename}")


#%%
