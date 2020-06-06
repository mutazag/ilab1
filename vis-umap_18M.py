# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # C33 UMAP Visualisation of 18M 
# 
# UMAP for 18M
# %% [markdown]
# ## Imports

# %%
import sys 
import os
import pathlib
import time
print(sys.version)


# %%
utils_path = pathlib.Path(os.getcwd() + '/utils')  # i suspect this one is not needed
print(utils_path.exists())
print(os.getcwd())
#sys.path.append(str(utils_path))  # may not be necessary
#sys.path.append(os.getcwd())  # i thnk this is the one that works 
sys.path.append('../') # this one is one level up so we can see the utils lib
print(sys.path)


# %%
import numpy as np
import sklearn
from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils.data import Data
from utils.config import Config


# %%
import umap
import numba

# %% [markdown]
# ## Read Data

# %%
d = Data()
df = d.get18M_features(descriptors='C33')

## need to add commenserate and DFT here 


# %%
df.shape


# %%
df.info(memory_usage='deep')

# ## umap_fit function
# call with df_features[feature_cols]
# labels_df has two columns C33 and communsurate 

def umap_fit(df,
             plot_df_filename, 
             index_name='uid', 
             n_neighbors=15, 
             n_components=2, 
             min_dist=0.1, 
             metric='euclidean'):

    reducer = umap.UMAP(n_neighbors=n_neighbors,
                        n_components=n_components,
                        min_dist=min_dist,
                        metric=metric,
                        random_state=50)

    embeddings = reducer.fit_transform(df)
    embeddings_df = pd.DataFrame(embeddings, columns={'x', 'y'})

    embeddings_df.set_index(index_name, inplace=True)

    embeddings_df.to_csv(Config().get_datapath(plot_df_filename))

    return(Config().get_datapath(plot_df_filename))

# %% [markdown]
# # Generate Umap plot dfs 

# %%


n = df.shape[0]
neighbors_list = [30]
components = 2

filename_pattern = 'umap_18Mdf_{}_{}_{}_{}.csv'


# %%
for neighbors in neighbors_list: 
    time_start = time.time()
    filename = filename_pattern.format('C33', n, neighbors, components)
    print(f'prep for filename: {filename}')
    plot_df = umap_fit(
        df=df,
        plot_df_filename=filename,
        n_neighbors=neighbors,
        n_components=components)
    print('for neighbors: {}. {:.3f} secs'.format(neighbors, time.time() - time_start))
