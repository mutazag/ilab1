

bilayer=['Ta4Ni4Te8_Ta4Ni4Te8','Ta4Te10Pd6_Hf1N2','Ta2Ni4Te4_Ta4Ni4Te8','Ta2Ni4Te2Se2_Ta4Ni4Te8','Ta4Te10Pd6_Ta4Te10Pd6','Ta4Ni4Te8_Ta2Ni4Te6']
mono1= ['Ta4Ni4Te8','Ta4Te10Pd6', 'Ta2Ni4Te4', 'Ta2Ni4Te2Se2', 'Ta4Te10Pd6', 'Ta4Ni4Te8']
mono2=['Ta4Ni4Te8', 'Hf1N2', 'Ta4Ni4Te8', 'Ta4Ni4Te8', 'Ta4Te10Pd6', 'Ta2Ni4Te6']


import pandas as pd 
import numpy as np 

df = pd.DataFrame({
    "bilayer":bilayer,
    "mono1": mono1, 
    "mono2":mono2})
print(df)


def uid(row): 
    # print(type(row))
    # print(row[1:].sort_values().str.cat(sep='_'))
    # print(row.apply(list))
    # return row['mono1'] + row['mono2']
    return row[1:].sort_values().str.cat(sep='_')


print(df[['bilayer','mono1','mono2']].apply(uid, axis=1))
df['uid']=df.apply(uid, axis=1)

print(df)

print('END')