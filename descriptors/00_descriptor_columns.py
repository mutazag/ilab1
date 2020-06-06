#%% 
from utils.config import Config
import pandas as pd

#%% 
c = Config()

#%%
print('load descriptor files')
# A + C
desc_ie = pd.read_csv(c.descriptors_IE)
# B + C 
desc_c33 = pd.read_csv(c.descriptors_C33)

#%%
# Return a new Index with elements from the index that are not in other
# diff_columns is A 
print('find difference in columns')
diff_columns = pd.Series(desc_ie.columns.difference(desc_c33.columns))
ie_columns = pd.Series(desc_ie.columns)
c33_columns = pd.Series(desc_c33.columns)

print(f'ie_columns: {len(ie_columns)}\n' \
    f'c33_columns: {len(c33_columns)}\n'\
    f'diff_columns: {len(diff_columns)}')

#%%
# all columns is A + B + C
# c33_columns + difference
all_columns = c33_columns.append(diff_columns, ignore_index=True)

print(f'all_columns (dirty): {len(all_columns)}')

#%% 
# columns cleanup function 
def cleanup_columns(desc_columns, filename):
    f1 = desc_columns.str.lower().str.match('monolayer')
    f2 = desc_columns.str.lower().str.startswith('unnamed')
    f3 = desc_columns.str.lower().str.match('\\bc\\b')
    print(f'{filename}: filtering out: {f1.sum() + f2.sum() + f3.sum()} columns')
    desc_columns[f1 | f2 | f3]
    # remove unwanted columns using the comnbination of filters f1, f2,  and f3
    desc_columns = desc_columns[((f1 | f2 | f3) != True)]
    print(f'all_columns (clean): {len(desc_columns)}')
    print(f'save all_columns to file "{filename}""')
    desc_columns.to_csv(filename, header=False)


#%% 
# Create descriptor column files 
# 
cleanup_columns(c33_columns, c.descriptors_column_names_C33)
cleanup_columns(ie_columns, c.descriptors_column_names_IE)
cleanup_columns(all_columns, c.descriptors_column_names)
# #%%
# # cleanup all_columns by removing the columns named monolayer, unnamed*, and 'c'
# f1 = all_columns.str.lower().str.match('monolayer')
# f2 = all_columns.str.lower().str.startswith('unnamed')
# # f3 = all_columns.str.lower().str.match('^c$') # start ^ and end * 
# f3 = all_columns.str.lower().str.match('\\bc\\b')
# print(f'filtering out: {f1.sum() + f2.sum() + f3.sum()} columns')
# all_columns[f1 | f2 | f3]
# # remove unwanted columns using the comnbination of filters f1, f2,  and f3
# all_columns = all_columns[((f1 | f2 | f3) != True)]
# print(f'all_columns (clean): {len(all_columns)}')
# print(f'save all_columns to file "{c.descriptors_column_names}""')
# all_columns.to_csv(c.descriptors_column_names, header=False)

#%% [markdown]
# # Create Master Descriptors File


#%%
# index by monolayer 
desc_ie.set_index('Monolayer', inplace=True) 
desc_c33.set_index('Monolayer', inplace=True)

#%%
desc_ie[diff_columns]

#%%
desc_master = pd.concat([desc_c33, desc_ie[diff_columns]], axis=1)
desc_master = desc_master[all_columns]
#%%
# save master descriptors to file
desc_master.to_csv(c.descriptors_master)

#%%
