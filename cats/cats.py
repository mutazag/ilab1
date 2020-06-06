# %%
import pandas as pd
from pandas.api.types import union_categoricals
import numpy as np
from utils.config import Config

#%%
c = Config()
c.validate_files()

# %%
df = pd.read_csv(c.predicted_300K)

# %%
df = df[["Monolayer 1", "Monolayer 2"]]

# %%
cats1 = pd.unique(df.to_numpy().ravel())
print(f"size of cats1: {cats1.size}")
# %%


df2 = pd.read_csv(c.predicted_18M)

# %%
df2 = df2[['monolayer1', 'monolayer2']]

# %%
cats2 = pd.unique(df2.to_numpy().ravel())
print(f"size of cats2: {cats2.size:,}")
# %%
cats1_type = pd.CategoricalDtype(categories=cats1, ordered=False)
cats2_type = pd.CategoricalDtype(categories=cats2, ordered=False)

# %%
cats1 = pd.Categorical(cats1)
cats2 = pd.Categorical(cats2)
cats_union = union_categoricals([cats1, cats2]).unique()
cats_union.sort_values(inplace=True)
print(f"size of cats union: {cats_union.size:,}")

# %%
# save cats to file
catsdf = pd.DataFrame({
    "codes": cats_union.codes},
    index=cats_union.categories.to_series(),
)
catsdf.index.names = ['categories']

catsdf.to_csv(c.layer_categories)
