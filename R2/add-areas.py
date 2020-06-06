#%%
from os import listdir
import pandas as pd
import numpy as np
from numpy import linalg as LA
import re
from pathlib import Path


#%%
def calculate_area(path): 
    f = open(path,'r')
    lines = f.readlines()
    if len(lines) > 4:
        wanted_lines = [3,4]
        cmpt = 1
        #vector1
        x = np.array(re.findall("\d+\.\d+", lines[2]))
        vector1 = x.astype(np.float)

        #vector2
        x = np.array(re.findall("\d+\.\d+", lines[3]))
        vector2 = x.astype(np.float)


        v = np.cross(vector1, vector2)
        area = LA.norm(v)
        return area


#%%
calculate_area('./R2/CONTCARs/Ag1Bi1P2Se6-Zr1Cl2/CONTCAR') 

#%%
i=0
dirctories = listdir(r'./R2/CONTCARs')
areas = {}

#%%
# area calculation
for dirctory in dirctories:
    path = './R2/CONTCARs/'+dirctory+'/CONTCAR'
    my_file = Path(path)
    if my_file.is_file():
        areas[dirctory] = calculate_area(path)
        i += 1
#%%
#areas
df = pd.DataFrame.from_dict(areas, orient='index', columns=['Area'])
df.index.name = 'Bilayer'
df = df.reset_index()
df['Name'] = df['Bilayer'].apply(
    lambda x: x.replace('-T-', '-T_')
    if x.find('-T-') != -1
    else x.replace('-', '_', 1))
# %%
df[['Name', 'Area']].to_csv('./R2/areas.csv',index=False)

# %%
