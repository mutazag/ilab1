# %%
#
# # https://www.tjansson.dk/2018/04/parallel-processing-pandas-dataframes/
from utils.data import Data
import tqdm
import time
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import concurrent.futures
import multiprocessing


# %%
def read_split(file, col_names=None, header=None):
    """read a single csv file from a collection of file splits, to use this
    correctly all files should have the same structure, ideally should be
    generated using the split command

    Arguments:
        file {string} -- file path for the part to read

    Keyword Arguments:
        col_names {list} -- names of columns, best used with later split reads to
                            apply header (default: {None})
        header {None or 0} -- use 0 when reading the first file that contains
                            header and None for all other files (default: {None})

    Returns:
        pandas DataFrame -- dataframe result from file
    """
    return pd.read_csv(file, names=col_names, header=header)


def filename_components(x):
    """ returns a series of tuples that unpacks the filepath name to its coponents

    Arguments:
        x {pahtlib.Path} -- a Path object

    Returns:
        tuple -- tuple of filename, file suffix, file prefix
    """
    fpath = Path(x)
    return (fpath.name, fpath.stem, fpath.suffix)


# %% [markdown]
# ## list of files
# using the glob library, this code will get a list of files that match a specified pattern
# these files should have been prepared using the split command as follows
#
# ```
# mkdir split
# split -l 100 -d --additional-suffix=.csv --suffix-length=5  lasso_monolayer_data_C33.csv \
# ./split/lasso_monolayer_data_C33_
# wc -l split/*
# ```
#
# It is important that the outout filename includes  a sperator so that the numeric suffix is
# not confused with any numerics in the file name, in this example I used an underscore _ as
# the seperator. Output file name will be similar to:  lasso_monolayer_data_C33_00000.csv
# this ensures that we can extract the sequence number from the file name and use for indexing
# and determining the partial dataframes an find the first file in the list to extract the
# header names

# %%
relative_path = './data/ML_IE_C33/18M_uid_split/*.csv'
# relative_path = './descriptors/split/*.csv'
files_glob = glob.glob(relative_path)
df_glob = pd.DataFrame(files_glob, columns=['fn'])
# %%

fcomponents_tuple_series = df_glob.fn.apply(filename_components)
fcomponents_colns = fcomponents_tuple_series.apply(
    pd.Series, index=['fname', 'stem', 'suffix'])
# update df glob with filename components
df_glob = pd.concat([df_glob, fcomponents_colns], axis=1)

# %% [markdown]
# so far the df_glob contains the following:
# fn: file path, to be used later to load files
# fname: file name only
# stem: the name of the file with out the extension
# suffix: the file extension
# example output for one entry:
#
# ```
# > df_glob.iloc[0]
# fn        ./descriptors/split/lasso_monolayer_data_C33_0...
# fname                    lasso_monolayer_data_C33_00006.csv
# stem                         lasso_monolayer_data_C33_00006
# suffix                                                 .csv
# Name: 0, dtype: object
# ```

# %%
# df_glob.stem.str.split('^(?P<idx>\d+)')
# (?P<idx>\d+)$
# df_glob.stem.str.split('(\d+)$').head() # this will split when it finds a number at the end

# df_glob['sss'] = "18M_as_as000"
# df_glob['sss'].str.split('(\d+)').head() # this will split everytime it finds a number

# what i need is not a split, it should be a search

# %% [markdown]
# This part extracts the file name from the stem column, file number represents the sequence
# of the partial file in the split set.
#
# `df_glob` is then indexed and sorted using the file number

# %%
# use a regex expression to extract the last sequence of digits in the file stem name
df_glob['file_number'] = df_glob.stem.str.extract('(\d+)$').astype(int)
# index using the extracted file number and sort
df_glob.set_index('file_number', inplace=True)
df_glob.sort_index(inplace=True)

# %% [markdown]
# # Read dataframe in parts
#
# the process invovles the following steps:
# 1. prepare an empty list to hold partial dataframes
# 2. read the first file in the list of splitted files, infer header since the first file already includes a header. \
# append file to the list of data frames.
# 3. loop to read all remaining files in the list of splitted files, specify that the files dont include header, \
# get header column names from the column names of the first dataframe, append part to the list of data frames
# 4. concat the list of dataframes to produce the final dataframe

# %%
# first file contains the header, rest of files have no header
first_file = df_glob.fn[0]
rest_of_files = df_glob.fn[1:]

# %% [markdown]
# ## 1. read files without multiprocessing

# %%
# prepare an empty list for storing the dataframes imported from files
dfs1 = []
print(f"dfs list is empty: {len(dfs1)}")

# read and append the first file to the data frames list dfs
timer1_read_start = time.time()
dfs1.append(read_split(first_file, header=0))
print(f"first is file with header, number of rows = {dfs1[0].shape[0]}")

# using list comprehension to read the remaining splits, applying header form the first file
print("subsequent reads will use the header from first df")
[dfs1.append(read_split(fn, dfs1[0].columns))
 for fn in tqdm.tqdm(rest_of_files)]

timer1_read_end = time.time()
timer1_read = timer1_read_end - timer1_read_start
print(f"time to read files {timer1_read:.2f}")
print(f"appended dataframes to list dfs1, size of list: {len(dfs1)}")
# %%
timer1_concat_start = time.time()
full_df1 = pd.concat(dfs1, ignore_index=True)  # , names=dfs[0].columns)
timer1_concat_end = time.time()
timer1_concat = timer1_concat_end - timer1_concat_start
print(f"time to concat dfs  {timer1_read:.2f}")
print(f"size of full df: {full_df1.shape[0]}")
# %%


# %%


# %% [markdown]
# ## 2. read files WITH multiprocessing

# %%
cpu_count = multiprocessing.cpu_count()
timer2_read_start = time.time()

dfs2_first = read_split(first_file, header=0)

# %%
with concurrent.futures.ProcessPoolExecutor(cpu_count) as pool:
    # [df2.append(read_split(fn, dfs2[0].columns)) for fn in rest_of_files]
    # df['result'] = list(tqdm.tqdm(pool.map(func, df['a'], df['b'], chunksize=10), total=df.shape[0]))
    # # With a progress bar

    df_list = list(tqdm.tqdm(
        pool.map(read_split, rest_of_files),
        total=len(rest_of_files)))

    df_combined = pd.concat(df_list)
    df_combined.columns = dfs2_first.columns

dfs2_full = dfs2_first.append(df_combined)


timer2_read_end = time.time()
timer2_read = timer2_read_end - timer2_read_start
print(f'\ndfs2_full shape = {dfs2_full.shape}')
print(f'\ntime to read and concat files {timer2_read:.2f}')

# %% [markdown]
# ## 3. import as a single file using pd.read_csv

# %%
timer3_read_start = time.time()
df3 = Data().get18M()
timer3_read = time.time() - timer3_read_start
# %%
print(f'normal import {timer1_read + timer1_concat:.2f}')

print(f'parallel import {timer2_read:.2f}')

print(f'pandas.read_csv {timer3_read:.2f}')

# %%
