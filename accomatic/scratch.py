import pandas as pd 
import matplotlib.pyplot as plt 
import sys



fields1 = ['Date12[DDMMYYYYhhmm]', 'Tair[C]','snow_depth[mm]']
fields2 = ['Date12[DDMMYYYYhhmm]', '100.000000 ']

pth = '/home/hma000/accomatic-web/tests/test_data/csvs/snow/s_%s'
pth2 = '/home/hma000/accomatic-web/tests/test_data/csvs/snow/g_%s'

m = 'merra.txt'
ms = pd.read_csv(pth % m, usecols=fields1, index_col=0, parse_dates=True).resample('M').mean()
mg = pd.read_csv(pth2 % m, usecols=fields2, index_col=0, parse_dates=True).resample('D').mean()

e = 'era.txt'
es = pd.read_csv(pth % e, usecols=fields1, index_col=0, parse_dates=True).resample('M').mean()
eg = pd.read_csv(pth2 % e, usecols=fields2, index_col=0, parse_dates=True).resample('D').mean()

j = 'jra.txt'
js = pd.read_csv(pth % j, usecols=fields1, index_col=0, parse_dates=True).resample('M').mean()
jg = pd.read_csv(pth2 % j, usecols=fields2, index_col=0, parse_dates=True).resample('D').mean()

### SNOW DEPTH ###
plt.rcParams["font.size"] = "16"
fig, ax = plt.subplots(figsize=(15, 6))
for df, lab in zip([ms, js, es], ['MERRA','JRA-55','ERA5']):
    #df = df[df['snow_depth[mm]'] != 0]
    df['snow_depth[mm]'] = df['snow_depth[mm]'] / 10
    plt.plot(df.index, df['snow_depth[mm]'], label=lab)
ax.set(xlabel="Date", ylabel="Snow Depth (cm)")
plt.legend()
plt.savefig(pth % 'plot.png')

### AIR TEMP AND GST ###

plt.rcParams["font.size"] = "16"
fig, ax = plt.subplots(figsize=(15, 6))
for df, lab in zip([ms, js, es], ['MERRA','JRA-55','ERA5']):
    df = df[df['snow_depth[mm]'] != 0]
    plt.plot(df.index, df['snow_depth[mm]'], label=lab)
ax.set(xlabel="Date", ylabel="Snow Depth")
plt.legend()
plt.savefig(pth % 'plot.png')