#%%
import re
import pandas as pd 

#%%
# processing bilayer and monolayer energies
bilayers_filename = './R2/BilayersEnergies'
monolayers_filename = './R2/MonolayersEnergies'
with open(bilayers_filename, 'r') as ofile: 
    bi_lines = ofile.readlines()

with open(monolayers_filename, 'r') as ofile: 
    mo_lines = ofile.readlines()

# print(lines)

#%%
#parse the energes files 


def energies_parse_lines(lines):
    pattern0 = '(?P<Name>.*)(\/$)'
    pattern1 = '(?P<Value>[+-]?[0-9]*[.]?[0-9]*)(?: eV$)'

    results = []
    #%%
    for i in range(int(len(lines) / 2)):
        print(i)
        print(lines[i])
        m0 = re.search(pattern0, lines[i])
        if m0: 
            name = m0.group('Name')
        print(lines[i + 1])
        m1 = re.search(pattern1, lines[i + 1])
        if m1: 
            value = m1.group('Value')

        if m0 and m1: 
            results.append([name, float(value)])

    df = pd.DataFrame(results, columns=['Name', 'Value'])
    return df


# %%
bilayers_df = energies_parse_lines(bi_lines)
bilayers_df['Name'] = bilayers_df['Name'].apply(
    lambda x: x.replace('-T-', '-T_')
    if x.find('-T-') != -1
    else x.replace('-', '_', 1))
mono_df = energies_parse_lines(mo_lines)


# %%
# process the count of monolayers in bilayer 
# from file IE_validationm_set 

countmonolayers_file = './R2/IE_validation_set'
with open(countmonolayers_file, 'r') as ofile: 
    countmonolayers_lines = ofile.readlines()

# %%


def counts_partse_lines(lines):
    pattern0 = '(?P<Name>.*)(?:-selected.dat\n$)'
    pattern1 = '(?P<m1_count>[0-9]*)[_](?P<m2_count>[0-9]*)(?:\n$)'

    results = []
    #%%
    for i in range(int(len(lines) / 2)):
        print(i)
        print(lines[i])
        m0 = re.search(pattern0, lines[i])
        if m0: 
            name = m0.group('Name')
        print(lines[i + 1])
        m1 = re.search(pattern1, lines[i + 1])
        if m1: 
            value1 = m1.group('m1_count')
            value2 = m1.group('m2_count')

        if m0 and m1 and value1.isnumeric() and value1.isnumeric(): 
            results.append([name, int(value1), int(value2)])

    df = pd.DataFrame(results, columns=['Name', 'Count_m1', 'Count_m2'])
    return df

    

# %%
counts_df = counts_partse_lines(countmonolayers_lines)
counts_df['Name'] = counts_df['Name'].apply(
    lambda x: x.replace('-T-', '-T_')
    if x.find('-T-') != -1
    else x.replace('-', '_', 1))
# %%


#%% 
# combine bilyaer energies with counts of monolayers 
# 
df = bilayers_df
df[['monolayer1', 'monolayer2']] = df.Name.str.split('_', expand=True)
#%%
df = df.iloc[:,[0,2,3,1]].merge(counts_df, how='inner', on='Name')


# %%


# %%
df = df.merge(mono_df.rename( 
        columns={ 
            'Name':'monolayer1',
            'Value':'ie_m1'}
            ), 
        how='inner', 
        on='monolayer1', 
        suffixes={'_df', '_m1'}) \
    .merge(mono_df.rename( 
        columns={ 
            'Name':'monolayer2',
            'Value':'ie_m2'}
            ), 
        how='inner', 
        on='monolayer2', 
        suffixes={'_df', '_m2'})

# %%
areas_df = pd.read_csv('./R2/areas.csv')

# %%
df.merge(
    areas_df.rename(columns={'Bilayer':'Name'}),
    how='inner', 
    on='Name') 


# %%
