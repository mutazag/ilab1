
import pandas as pd
import numpy as np
import time
from utils.config import Config


class LayerCodes:
    __catType = 0

    def __init__(self):
        c = Config()
        df = pd.read_csv(c.layer_categories, index_col="categories")
        # cat = pd.Categorical.from_codes(
        #     codes=df.codes,
        #     categories=df.index)
        self.__catType = pd.CategoricalDtype(categories=df.index)

    @property
    def LayersCategoricalType(self):
        return self.__catType


def uid(row):
    """Generate a unique ID for monolayer combinations in the formate of bilayer name

    usage:
        df['uid']=df[['bilayer','mono1','mono2']].apply(uid, axis=1)
    Arguments:
        row {DataFrame} -- Data frame raw must have three columns: bilayer name, monolayer 1 name, monolayer 2 name
    Returns:
        string -- new unique ID based on monolayer names
    """
    return row[1:].sort_values().str.cat(sep='_')


def add_uid(filename, columns, outfilename):
    """add uuid to dataset

    Arguments:
        filename {string} -- input file namefilename
        columns {string} -- list of column names: bilayer name, monolayer 1 name, monolayer 2 name 
            --['bilayer','monolayer1','monolayer2']
        outfilename {string} -- output filename
    """
    timer = time.time()
    print(f"read file: {filename}")
    df = pd.read_csv(filename)
    print(f"timer - read file: {time.time() - timer}")

    timer = time.time()
    print(f"add uid to columns {columns}")
    df['uid'] = df[columns].apply(uid, axis=1)
    print(f"timer - add uid using method 1: {time.time() - timer}")

    timer = time.time()
    print(f"save to file: {outfilename}")
    df.to_csv(outfilename)
    print(f"timer - savefile: {time.time() - timer}")

    return outfilename


def add_uid2(filename, columns, outfilename):
    """add uuid to dataset - new method to speed up processing time

    Arguments:
        filename {string} -- input file namefilename
        columns {string} -- list of column names: bilayer name, monolayer 1 name, monolayer 2 name 
            --['bilayer','monolayer1','monolayer2']
        outfilename {string} -- output filename
    """
    timer = time.time()
    print(f"read file: {filename}")
    df = pd.read_csv(filename)
    print(f"timer - read file: {time.time() - timer}")

    timer = time.time()
    print(f"add uid to columns {columns}")

    uid_arr = df[columns[1:]].apply(lambda x: [x[0], x[1]], axis=1)
    uid_arr_sorted = uid_arr.apply(lambda x: np.sort(x))
    df['uid'] = uid_arr_sorted.apply('_'.join)
    print(f"timer - add uid using method 2: {time.time() - timer}")

    timer = time.time()
    print(f"save to file: {outfilename}")
    df.to_csv(outfilename)
    print(f"timer - savefile: {time.time() - timer}")

    return outfilename
