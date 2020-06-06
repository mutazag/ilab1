#%% 
# 
# # https://www.tjansson.dk/2018/04/parallel-processing-pandas-dataframes/ 
import tqdm   
import time                                                                                                
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import concurrent.futures
import multiprocessing
cpu_count = multiprocessing.cpu_count()
 
#%% 
def read_split(file, col_names=None, header=None):
    return pd.read_csv(file, names=col_names, header=header)
#%% 
df = read_split('./descriptors/split/f01.csv')

print(df.head())
print(df.shape)
print(df.memory_usage())

#%%
df2 = read_split('./descriptors/split/f00.csv', header=0)
print(df2.head())
print(df2.shape)
print(df2.memory_usage())

#%% 
info = df.info()
print(df.info())


#%% 
files_glob = glob.glob('./descriptors/split/*.csv')
df_glob = pd.DataFrame(files_glob, columns=['fn'])
# df_glob['filename'] = df_glob.fn.apply(lambda x: Path(x).name)
#%%

def filename_components(x): 
    """ returns a series of tuples that unpacks the filepath name to its coponents
    
    Arguments:
        x {[type]} -- [description]
    
    Returns:
        tuple -- tuple of filename, file suffix, file prefix
    """
    fpath = Path(x)
    return (fpath.name, fpath.stem, fpath.suffix)


fcomponents_tuple_series = df_glob.fn.apply(filename_components)
fcomponents_colns = fcomponents_tuple_series.apply(pd.Series,index=['fname','stem','suffix'])

#%%
#update df glob with filename components
df_glob = pd.concat([df_glob, fcomponents_colns], axis=1)

#%%
# df_glob.stem.str.split('^(?P<idx>\d+)')
# (?P<idx>\d+)$
# df_glob.stem.str.split('(\d+)$').head() # this will split when it finds a number at the end

# df_glob['sss'] = "18M_as_as000"
# df_glob['sss'].str.split('(\d+)').head() # this will split everytime it finds a number

# what i need is not a split, it should be a search 

#%%
df_glob['file_number'] = df_glob.stem.str.extract('(\d+)$').astype(int)


#%%
df_glob.set_index('file_number', inplace=True)


#%%
df_glob.sort_index(inplace=True)

#%%
print(df_glob.size)
print(df_glob.shape)

#%%
first_file = df_glob.fn[0]
rest_of_files = df_glob.fn[1:]

#%%

dfs = [] 
print(f"dfs list is empty: {len(dfs)}")
#%%
dfs.append(read_split(first_file, header=0))
print(f"firsts file with header, number of rows = {dfs[0].shape[0]}")

#%%
print("subsequent reads will use the header frin furst df")
[dfs.append(read_split(fn, dfs[0].columns)) for fn in rest_of_files]



#%%
print(f"appended dfs to a list, size of list: {len(dfs)}")
[print(dff.shape[0]) for dff in dfs]
print(f"total number of rows: {sum([dff.shape[0] for dff in dfs])}")


#%% 
dfs[0].head()
#%%
dfs[1].head()


#%%
full_df = pd.concat(dfs, ignore_index=True)  # , names=dfs[0].columns)

#%%
print(f"size of full df: {full_df.shape[0]}")

#%%
