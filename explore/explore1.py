from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt 


datafolder = Path("data/ML_IE_C33")
print(datafolder)

file1 = datafolder  / "300k_PREDICTED.csv"

print(file1.exists())

df = pd.read_csv(file1)

df.info()
df.head()

print(df.columns)

df2 = df[["Monolayer 1", "Monolayer 2","C33 (GPa)", "IE (J/m^2)" ]]

df2.plot()
plt.show()

df2.plot.bar()
plt.show()

plt.subplot(2,1,1)
df2['Monolayer 1'].value_counts().plot( kind='bar')
plt.subplot(2,1,2)
df2['Monolayer 2'].value_counts().plot( kind='bar')
plt.show()

df2.groupby('Monolayer 1').nunique()
df2['Monolayer 1'].nunique()
df2['Monolayer 1'].value_counts()
df2['Monolayer 2'].value_counts()


