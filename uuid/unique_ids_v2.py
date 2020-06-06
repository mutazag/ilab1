import numpy as np
import pandas as pd
import time

bilayer = ['Ta4Ni4Te8_Ta4Ni4Te8', 'Ta4Te10Pd6_Hf1N2', 'Hf1N2_Ta4Te10Pd6', 'Ta2Ni4Te4_Ta4Ni4Te8',
           'Ta2Ni4Te2Se2_Ta4Ni4Te8', 'Ta4Te10Pd6_Ta4Te10Pd6', 'Ta4Ni4Te8_Ta2Ni4Te6']
mono1 = ['Ta4Ni4Te8', 'Ta4Te10Pd6', 'Hf1N2', 'Ta2Ni4Te4',
         'Ta2Ni4Te2Se2', 'Ta4Te10Pd6', 'Ta4Ni4Te8']
mono2 = ['Ta4Ni4Te8', 'Hf1N2', 'Ta4Te10Pd6',
         'Ta4Ni4Te8', 'Ta4Ni4Te8', 'Ta4Te10Pd6', 'Ta2Ni4Te6']


df = pd.DataFrame({
    "bilayer": bilayer,
    "mono1": mono1,
    "mono2": mono2})
print(df)


def uid(row):
    # print(type(row))
    # print(row[1:].sort_values().str.cat(sep='_'))
    # print(row.apply(list))
    # return row['mono1'] + row['mono2']
    return row[1:].sort_values().str.cat(sep='_')

timer = time.time()
print(df[['bilayer', 'mono1', 'mono2']].apply(uid, axis=1))
df['uid'] = df.apply(uid, axis=1)
print(f"method 1 time: {time.time() - timer}")


timer = time.time()
df['uid_arr'] = df[['mono1', 'mono2']].apply(lambda x: [x[0], x[1]], axis=1)
df['uid_arr_sorted'] = df.uid_arr.apply(lambda x: np.sort(x))
df['uid_new'] = df.uid_arr_sorted.apply('_'.join)
print(f"method 2 time: {time.time() - timer}")


print(df)

print('END')
