# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


class Config:
    """Config class encapsulate operations to manage folder paths for data sets
    """
    __data_dir = ""
    __predicted_300K = "300k_PREDICTED.csv"
    __predicted_18M = "18M_PREDICTED.csv"
    __small_300K = "300K_small.csv"
    __small_18M = "18M_small.csv"
    __uid_300K = "300K_uid_2.csv"
    __uid_18M = "18M_uid_2.csv" # update name
    __features_300K = "300K_full_features.csv"
    __features_300K_v2 = "300K_full_features_conflag.csv"
    __features_300K_v3 = "300K_full_features_flags.csv"
    __features_18M = "18M_full_features.csv"
    __private = "private"
    __layercategories = "layercategories.csv"
    __descriptors_dir = ""
    __descriptors_IE = "lasso_monolayer_data_IE.csv"
    __descriptors_C33 = "lasso_monolayer_data_C33.csv"
    __descriptors_column_names = 'all_descriptors300K_columns.csv'
    __descriptors_column_names_C33 = 'all_descriptors300K_columns_C33.csv'
    __descriptors_column_names_IE = 'all_descriptors300K_columns_IE.csv'
    __descriptors_master = 'descriptors_master.csv'
    __descriptors_master_6k = 'descriptors_master_6k.csv'
    __con_bilayers = 'con_bilayers.csv'
    __dft_C33_uid = 'C33_DFT_uid.csv'
    __dft_IE_uid = 'IE_DFT_uid.csv'

    def __init__(self, datafolder="data/ML_IE_C33", descriptorsfolder="descriptors"):
        self.__data_dir = Path(datafolder)
        self.__descriptors_dir = Path(descriptorsfolder)

    def __get_file_18M(self, small=True):
        if (small):
            return self.__data_dir / self.__small_18M
        else:
            return self.__data_dir / self.__predicted_18M

    def __set_file_18M(self, filename, small=True):
        if (small):
            self.__small_18M = filename
        else:
            self.__predicted_18M = filename

    def __get_file_300K(self, small=True):
        if (small):
            return self.__data_dir / self.__small_300K
        else:
            return self.__data_dir / self.__predicted_300K

    def __set_file_300K(self, filename, small=True):
        if (small):
            self.__small_300K = filename
        else:
            self.__predicted_300K = filename

    @property
    def predicted_18M(self):
        return self.__get_file_18M(small=False)

    @predicted_18M.setter
    def predicted_18M(self, filename):
        self.__set_file_18M(filename, small=False)

    @property
    def small_18M(self):
        return self.__get_file_18M(small=True)

    @small_18M.setter
    def small_18M(self, filename):
        self.__set_file_18M(filename, small=True)

    @property
    def uid_18M(self):
        return self.__data_dir / self.__uid_18M

    @uid_18M.setter
    def uid_18M(self, filename):
        self.__uid_18M = filename

    @property
    def predicted_300K(self):
        return self.__get_file_300K(small=False)

    @predicted_300K.setter
    def predicted_300K(self, filename):
        self.__set_file_300K(filename, small=False)

    @property
    def small_300K(self):
        return self.__get_file_300K(small=True)

    @small_300K.setter
    def small_300K(self, filename):
        self.__set_file_300K(filename, small=True)

    @property
    def uid_300K(self):
        return self.__data_dir / self.__uid_300K

    @uid_300K.setter
    def uid_300K(self, filename):
        self.__uid_300K = filename

    @property
    def features_300K(self):
        return self.__data_dir / self.__features_300K

    @property
    def features_300K_v2(self):
        return self.__data_dir / self.__features_300K_v2

    @property
    def features_300K_v3(self):
        return self.__data_dir / self.__features_300K_v3

    @property
    def features_18M(self):
        return self.__data_dir / self.__features_18M

    @property
    def layer_categories(self):
        return self.__data_dir / self.__layercategories

    @layer_categories.setter
    def layer_categories(self, filename):
        self.__layercategories = filename

    @property
    def descriptors_IE(self):
        return self.get_descriptorspath(self.__descriptors_IE)

    @property
    def descriptors_C33(self):
        return self.get_descriptorspath(self.__descriptors_C33)

    @property
    def descriptors_master(self):
        return self.get_descriptorspath(self.__descriptors_master)

    @property
    def descriptors_master_6k(self):
        return self.get_descriptorspath(self.__descriptors_master_6k)

    @property
    def descriptors_column_names(self):
        return self.get_descriptorspath(self.__descriptors_column_names)

    @property
    def descriptors_column_names_C33(self):
        return self.get_descriptorspath(self.__descriptors_column_names_C33)

    @property
    def descriptors_column_names_IE(self):
        return self.get_descriptorspath(self.__descriptors_column_names_IE)

    @property
    def con_bilayers(self):
        return self.get_descriptorspath(self.__con_bilayers)

    @property
    def dft_C33_uid(self):
        return self.get_descriptorspath(self.__dft_C33_uid)

    @property
    def dft_IE_uid(self):
        return self.get_descriptorspath(self.__dft_IE_uid)

    def get_datapath(self, filename):
        return self.__data_dir / filename

    def get_descriptorspath(self, filename):
        return self.__descriptors_dir / filename

    def validate_files(self):
        filetype = ["predicted", "small", "uid", "predicted",
                    "small", "uid", "reference", "descriptors", "descriptors"]
        fileorigin = ["18M", "18M", "18M", "300K", "300K",
                      "300K", "reference", "descriptors", "descriptors"]
        filenames = [
            self.__predicted_18M,
            self.__small_18M,
            self.__uid_18M,
            self.__predicted_300K,
            self.__small_300K,
            self.__uid_300K,
            self.__layercategories,
            self.__descriptors_IE,
            self.__descriptors_C33]

        filepaths = [
            self.predicted_18M,
            self.small_18M,
            self.uid_18M,
            self.predicted_300K,
            self.small_300K,
            self.__uid_300K,
            self.layer_categories,
            self.descriptors_IE,
            self.descriptors_C33]

        filevalidation = [
            self.predicted_18M.exists(),
            self.small_18M.exists(),
            self.uid_18M.exists(),
            self.predicted_300K.exists(),
            self.small_300K.exists(),
            self.uid_300K.exists(),
            self.layer_categories.exists(),
            self.descriptors_IE.exists(),
            self.descriptors_C33.exists()]

        validation = {
            "filename": filenames,
            "filepaths": filepaths,
            "fileexists": filevalidation
        }

        index = pd.MultiIndex.from_arrays(
            [filetype, fileorigin], names=["type", "origin"])
        return pd.DataFrame(validation, index=index)
