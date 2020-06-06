#%% 
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt 

#%%
datafolder = Path("data/ML_IE_C33")
print(datafolder)

#%%
file1 = datafolder  / "300k_PREDICTED.csv"
file2 = datafolder / "18M_PREDICTED.csv"

print(file1.exists())
print(file2.exists())

#%%
df1 = pd.read_csv(file1)

df1.info()
df1.head()

print(df1.columns)

#%%
df1 = df1[["bilayer", "Monolayer 1", "Monolayer 2","C33 (GPa)", "IE (J/m^2)" ]]


#%%
df2 = pd.read_csv(file2)
df2.info()
df2.head()
print(df2.columns)

#%%
df2 = df2[["bilayer", "monolayer1", "monolayer2", "C33 (GPa)", "IE (J/m^2)"]]
df2.info()

#%%
df2.to_csv(datafolder /"18M_small.csv", index=False)
df1.to_csv(datafolder /"300K_small.csv", index=False)



#%%
