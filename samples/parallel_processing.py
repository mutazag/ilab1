#!python3.6
# https://www.tjansson.dk/2018/04/parallel-processing-pandas-dataframes/ 
import tqdm                                                                                                   
import numpy as np
import pandas as pd
import concurrent.futures
import multiprocessing
num_processes = multiprocessing.cpu_count()
 
# Create a dataframe with 1000 rows
df = pd.DataFrame({i: np.random.randint(1,100,size=10000000) for i in ['a', 'b', 'c']})
 
# Define a function on the numbers
def func(a, b):
    return a+b
 
# Process the rows in chunks in parallel
with concurrent.futures.ProcessPoolExecutor(num_processes) as pool:
    #df['result'] = list(pool.map(func, df['a'], df['b'], chunksize=10)) # Without a progressbar
    df['result'] = list(tqdm.tqdm(pool.map(func, df['a'], df['b'], chunksize=10), total=df.shape[0])) # With a progressbar