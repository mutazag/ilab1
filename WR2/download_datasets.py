#%%
from __future__ import absolute_import
import os
import sys

print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
# print(__doc__)
# print(f'current file: {__file__}')
# print(f'module name {__name__}')

#%%
print(f'current folder {os.getcwd()}')
print(f'sys path {sys.path}')

#%% 
#import utils.downloader]
# from ..utils.downloader import download_files

#%% 
# sys.path.append(os.getcwd())
print(sys.path)
# import data
from utils.data import Data
from utils.downloader import download_files, get_file_name_from_cd, get_file_name_from_resposne

#%%
download_files(['url'], './data/WR2')

print('END')


#%%
