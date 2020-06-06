import pandas as pd
import numpy as np
import time
from glob import glob
import tqdm
from utils.config import Config


class Data:
    __catType = 0
    __c = None
    __300K_columns_index = [15, 2, 3, 5, 7, 8, 9, 10, 11, 12]
    __300K_columns_labels = ['uid', 'bilayer', 'monolayer1', 'monolayer2',
                             'IE', 'IE_error', 'IE_rel_error', 'C33', 'C33_error', 'C33_rel_err']
    __18M_columns_index = [15, 4, 5, 6, 7, 8, 11, 9, 10, 12]
    __18M_columns_labels = ['uid', 'bilayer', 'monolayer1', 'monolayer2',
                            'IE', 'IE_error', 'IE_rel_error', 'C33', 'C33_error', 'C33_rel_err']

    def __init__(self):
        self.__c = Config()

    def get300K(self, indexed=True):
        timer = time.time()
        df = pd.read_csv(self.__c.uid_300K, low_memory=False)
        df = df.iloc[:, self.__300K_columns_index]
        df.columns = self.__300K_columns_labels

        if indexed is True:
            print('set index')
            df.set_index('uid', inplace=True)

        print(f"time to load {time.time() - timer :.2f}")
        return df

    def get300K_features(self, indexed=True, v2=False):
        timer = time.time()

        filename = self.__c.features_300K
        if v2 is True: 
            filename = self.__c.features_300K_v3

        df = pd.read_csv(filename, low_memory=False)

        if indexed is True:
            print('set index')
            df.set_index('uid', inplace=True)

        print(f"time to load {time.time() - timer :.2f}")
        return df

    @property
    def columns_index_18M(self): 
        return self.__18M_columns_index

    @property
    def columns_labels_18M(self):
        return self.__18M_columns_labels

    def get18M(self, indexed=True):
        timer = time.time()
        df = pd.read_csv(self.__c.uid_18M, low_memory=False)
        df = df.iloc[:, self.__18M_columns_index]
        df.columns = self.__18M_columns_labels

        if indexed is True:
            print('set index')
            df.set_index('uid', inplace=True)

        print(f"time to load {time.time() - timer :.2f}")
        return df

    def get_float_types(self, filename): 
        # inspired from: https://www.dataquest.io/blog/pandas-big-data/
        # ''Selecting Types While Reading the Data In''
        # read few rows, infer types, then change float types 
        df_temp = pd.read_csv(filename, nrows=5)
        # get types of columns 
        df_types = df_temp.dtypes
        # filter by float types only
        df_types = df_types[df_types == 'float64']  
        # create a dict {'col_name': 'float32', ...}
        read_float_types = dict(zip(df_types.index, ['float32' for i in df_types.values]))
        return read_float_types

    def get_features_df_columns(self, filename, float_dtypes, descriptors):
        df = pd.read_csv(filename, nrows=5, dtype=float_dtypes, low_memory=False)
        cols1 = df.columns[0:10].to_series()  # returns the bilayer columns 
        cols1 = df.columns[0:10].to_series()  # returns the bilayer columns 
        if descriptors == 'C33':
            cols2 = self.getDescriptorsColumnNames_C33()
        else: 
            cols2 = self.getDescriptorsColumnNames_IE()
        cols = cols1.append(cols2)
        return cols

    def get18M_features(self, descriptors='C33', indexed=True, v2=False):
        timer = time.time()

        # df = pd.read_csv(filename, low_memory=False)
        files_path = self.__c.get_datapath(f'18M_full_features/chunk_*.csv')
        files_glob = glob(files_path.as_posix())
        float_dtypes = self.get_float_types(files_glob[0])
        cols = self.get_features_df_columns(files_glob[0], float_dtypes, descriptors)

        dfs_list = []
        for fn in tqdm.tqdm(files_glob):
            df_chunk = pd.read_csv(fn, dtype=float_dtypes, low_memory=False)
            dfs_list.append(df_chunk[cols])

        df = pd.concat(dfs_list, ignore_index=True)
        print(f'pd.concat {len(dfs_list)} df chunks: {time.time() - timer :.2f}s')

        if indexed is True:
            df.set_index('uid', inplace=True)
            print(f'set index: {time.time() - timer :.2f}s')

        print(f"time to load {time.time() - timer :.2f}s")
        return df

    def getDescriptorsColumnNames(self): 
        column_names = pd.read_csv(
            self.__c.descriptors_column_names,
            header=None, squeeze=True, index_col=0) 
        return column_names

    def getDescriptorsColumnNames_C33(self): 
        column_names = pd.read_csv(
            self.__c.descriptors_column_names_C33,
            header=None, squeeze=True, index_col=0) 
        return column_names

    def getDescriptorsColumnNames_IE(self): 
        column_names = pd.read_csv(
            self.__c.descriptors_column_names_IE,
            header=None, squeeze=True, index_col=0) 
        return column_names

    def getDescriptorsMaster(self, indexed=True):
        timer = time.time()
        df = pd.read_csv(self.__c.descriptors_master, low_memory=False)  # , index_col='Monolayer')
        df = df.rename(columns={'Monolayer': 'monolayer'})
        if indexed is True: 
            print('set index')
            df.set_index('monolayer', inplace=True)
        print(f"time to load {time.time() - timer :.2f}")
        return df 

    def getDescriptorsMaster_6k(self, indexed=True):
        timer = time.time()
        df = pd.read_csv(self.__c.descriptors_master_6k, low_memory=False)  # , index_col='Monolayer')
        df = df.rename(columns={'Monolayer': 'monolayer'})
        if indexed is True: 
            print('set index')
            df.set_index('monolayer', inplace=True)
        print(f"time to load {time.time() - timer :.2f}")
        return df 

