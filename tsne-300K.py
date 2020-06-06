# %%
import pandas as pd
from pandas.api.types import union_categoricals
import numpy as np
from utils.config import Config
import matplotlib.pyplot as plt

#%%
c = Config()
c.validate_files()

#%% 
#prepare new columns,labels
df1_columns_index = [2,3,5,7,8,9,10,11,12,15]
df1_columns_labels = ['bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err','uid (index)']

df2_columns_index = [4,5,6,7,8,11,9,10,12,16]
df2_columns_labels = ['bilayer', 'monolayer1','monolayer2','IE','IE_error','IE_rel_error','C33','C33_error','C33_rel_err','uid (index)']

# %%
df1 = pd.read_csv(c.uid_300K) #, nrows=10000)
#%%
df1.index = df1.uid
df1 = df1.iloc[:, df1_columns_index[:-1]]
df1.columns = df1_columns_labels[:-1]


# %%
import time
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import fetch_mldata, fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

#%%
np.random.seed(42)
rndperm = np.random.permutation(df1.shape[0])
# %%
N = 10000
# df_subset = df1.iloc[rndperm[:N], :].copy()

df_subset = df1.copy()
feat_cols = ['IE', 'IE_error', 'IE_rel_error', 'C33', 'C33_error', 'C33_rel_err']
data_subset = df_subset[feat_cols].values


# %%
time_start = time.time()
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(data_subset)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


#%%
df_subset['tsne-one'] = tsne_results[:, 0]
df_subset['tsne-two'] = tsne_results[:, 1]

#%%
plt.figure(figsize=(16, 10))
sns.scatterplot(
    x="tsne-one", y="tsne-two",
    data=df_subset,
    alpha=0.3
)

#%%
# plt.figure(figsize=(16, 10))
# sns.scatterplot(
#     x="tsne-one", y="tsne-two",
#     data=df_subset,
#     legend="full",
#     hue="y",
#     palette=sns.color_palette("hls", 10),
#     alpha=0.3
# )
