from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt 


datafolder = Path("data/ML_IE_C33")
print(datafolder)

file1 = datafolder  / "300k_small.csv"
file2 = datafolder / "18M_small.csv"

print(file1.exists())
print(file2.exists())

df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)



# find bilayers in the 18M set that are also from the 300K set 
df_merge = pd.merge(
    df2, df1, 
    left_on=['monolayer1','monolayer2'],
    right_on=['Monolayer 1','Monolayer 2'],
    how='left', 
    suffixes=('18M', '300k'),
    indicator='source')

df_merge.source.value_counts()


df1_count = df1.shape[0]
df2_count = df2.shape[0]
df_merge_count = df_merge.shape[0]
common_count = df_merge.source.value_counts()['both']


msg = "300K count: {:,}, 18M count: {:,}, merge count: {:,}, common bilayers count = {:,}".format(
    df1_count, 
    df2_count, 
    df_merge_count, 
    common_count
)

print(msg)

common_filter = df_merge.source.str.contains("both")
df_merge[common_filter].head()



#%% [markdown]
# ## 1) common elemetns 
# Element1xnElement2xm

# n:m is the same -> same monotloyar 
# order of the name is irrelevant 
# E1nE2m == E2mE1n
# E14E22 == E18E24

# -T1 --> means a different compound 


# ## 2) printing of pdf
# bilayer idx, byname [["monolayer1", "monolayer2", "C33 (GPa)", "IE (J/m^2)"] + errors and relative errors -> pdf 

# ## 3) visualiy find one heatmap in the other 
                   

#%%
