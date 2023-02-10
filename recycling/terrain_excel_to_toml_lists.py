import pandas as pd
import numpy as np

df = pd.read_csv('/home/hma000/storage/terrain_exp/sites_and_paramaters_unformatted.csv', usecols=['site', 'TWI', 'ISO', 'VEG', 'SNOW', 'MAT'], dtype = str)
df['com'] = df.TWI + df.VEG + df.ISO + df.SNOW + df.MAT
print(df.com.value_counts())


# Create a script to read in site params excel, calculate permutations
# Only select those sites which are featured in the > 5 groups
# Create dictionary of list:terr_cluster then print dict.keys() and dict.values()
# to put in toml file to see how various terrain classes are doing. 


df = pd.read_excel("/home/hma000/storage/terrain_exp/master_parameter_doc.xlsx", header=1, usecols=['site','TWI','ISO','VEG','SNOW','MAT'], dtype=str)
df['com'] = df.TWI + df.VEG + df.ISO + df.SNOW + df.MAT
clusters = df.com.value_counts()
clusters = clusters[(clusters > 4)].index.tolist()
test_values = list(range(1, len(clusters)+1))
clusters = {clusters[i]: test_values[i] for i in range(len(clusters))}

for i, row in df.iterrows():
    if row.com in clusters.keys():
        row.com = clusters[row.com]
    else:
        row.com = np.nan


df = df.dropna().reset_index(drop = True)

names, clusts = df.site.to_list(), df.com.to_list()
new_clust = {names[i]: str(clusts[i]) for i in range(len(names))}
print(new_clust.values())