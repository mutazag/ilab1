# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils.config import Config
from utils.layercodes import LayerCodes

c = Config()
c.validate_files()
l = LayerCodes()
dt = l.LayersCategoricalType

# %%
typeMonoLayers = l.LayersCategoricalType

# %%
newdf1 = pd.read_csv(c.predicted_300K, dtype={
    "Monolayer 1": typeMonoLayers,
    "Monolayer 2": typeMonoLayers,
    "bilayer_index": np.int16,
    "X1": np.int16,
    "Y1": np.int16
})
# total size reduction by 25% from 31.7MB to 23.6 MB
# %%
newdf2 = pd.read_csv(c.predicted_18M, dtype={
    "Unnamed: 0": np.uint16,
    "Unnamed: 0.1": np.uint16,
    "index_bilayer": np.uint32,
    "FlagX1": np.uint16,
    "FlagY1": np.uint16,
    "monolayer1": typeMonoLayers,
    "monolayer2": typeMonoLayers
})

# newdf2 size is 1.3GB
# df2 size is 2 GB
# reduction by 35%

# %%
newdf1['Monolayer 1'] = newdf1['Monolayer 1'].cat.remove_unused_categories()
l1group = newdf1.groupby("Monolayer 1")
l1group.X1.count().sort_values(ascending=False)
# %%
newdf1['Monolayer 2'] = newdf1['Monolayer 2'].cat.remove_unused_categories()
l2group = newdf1.groupby("Monolayer 2")
l2group.X1.count().sort_values(ascending=False)

# %%
l1group.nunique().dropna().sort_values(
    axis=0, by="bilayer_index", ascending=False)
# %%
# newdf1.groupby("Monolayer 2").values_count()


# %%
# %%
def heatmap(x, y, size):
    fig, ax = plt.subplots(figsize=(15,15))

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    # ax.set_xticks([x_to_num[v] for v in x_labels])
    # ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    # ax.set_yticks([y_to_num[v] for v in y_labels])
    # ax.set_yticklabels(y_labels)

# %%
heatmap(
    x=newdf1['Monolayer 1'],
    y=newdf1['Monolayer 2'],
    size=newdf1['IE (J/m^2)'].abs()
)

# %%
