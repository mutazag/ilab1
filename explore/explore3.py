#%%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt 

#%%
datafolder = Path("data/ML_IE_C33")
print(datafolder)

file1 = datafolder  / "300K_small.csv"

print(file1.exists())

#%%
df = pd.read_csv(file1)

#%%
df.info()
df.head()

print(df.columns)


#%%
df.groupby('Monolayer 1').nunique()
df['Monolayer 1'].nunique()
#%%
layer1_freq = df['Monolayer 1'].value_counts()
layer2_freq = df['Monolayer 2'].value_counts()


#%%
layer1_freq.plot.bar()
plt.show()
#%%
layer1_freq.plot.bar()
plt.title("Monolayer 1 : compound freq")
plt.show()


#%%
layer1_freq.plot(title = 'Monolayer 1 : compound freq', kind='bar')
plt.show()
#%%
layer2_freq.plot(title = 'Monolayer 2 : compound freq', kind='bar')
plt.show()





#%%
# check indecies on the two series for layer1 and layer2
print(layer1_freq.index)
print(layer2_freq.index)

#%%
ll = layer1_freq + layer2_freq
print(ll.index)
#%%
ll.plot(title = 'Monolayer (col1 and col2) : compound freq', kind='bar')
plt.show()


#%%
# pick one compound in the middle of the histogram 
e1 = layer1_freq[layer1_freq == 356]
e1_name = e1.index[0]
print(f"name: {e1_name}")


#%%
# match compound from layer 1 freq
print(layer2_freq[e1_name])
print(layer2_freq[layer2_freq == 415])

#%%
# filter df where layer 1 or layer 2 equals e1_name 

condition = (df['Monolayer 1'] == e1_name) | (df['Monolayer 2'] == e1_name)
print(condition.sum())


#%% filter to this one compound
e1_df = df [condition]

e1_df.info()
e1_df.head()

#%%
e1_df.plot.scatter(x='C33 (GPa)', y='IE (J/m^2)')
plt.show()
#%% [markdown]
# # converting layer names to categorical values 

#%%
df.plot.scatter(x='C33 (GPa)', y='IE (J/m^2)')
plt.show()

#%%
