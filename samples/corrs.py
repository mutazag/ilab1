#%%
import pandas as pd
import numpy as np
#%%
df = pd.DataFrame( np.random.randn(10,2), columns=list('ab') )
df

#%%
#add some grouping
df['g'] = 'X'
df.loc[:5,'g'] = 'Y'
df.index = df.g
df = df.drop('g', axis=1)

#%%
#basic corr() behaviour 
df.corr() 

#%%
#corr by group 
df.groupby('g').corr()

#%%
#corr for a group 'X'
df.loc['X'].corr()

#%% 
# corr by change the order of the variables  for a group 
df.loc['X',['b','a']].corr()

#%% 
# corr for all groups, change order of variables 
df[['b','a']].groupby('g').corr()

#%%
