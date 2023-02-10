import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

try: mer.close()
except: pass


all_sites = False
if all_sites:
    mer = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/scaled/to_compare/NRML_merra.nc', engine="netcdf4")
    df = mer.to_dataframe()
    df = df.reset_index(drop=False)
    df = df.drop(columns=['latitude', 'longitude','crs','height','station'])
    df.time = pd.to_datetime(df['time'])
    df = df.set_index(['time'])
    df['station_name'] = df.station_name.str.decode("utf-8")
    l = df.station_name.unique()
    mer.close()

else: l = ['YK']


pths = glob.glob('/home/hma000/storage/merra_precip_test/scaled/to_compare/*.nc')
for site in l:
    fig = plt.subplots(figsize=(12,7))

    for data, col in zip(pths, plt.cm.rainbow(np.linspace(0, 1, len(pths)))):
        
        if os.path.basename(data) == '09FEB_merra.nc': continue
        mer = xr.open_mfdataset(data, engine="netcdf4")
        df = mer.to_dataframe()
        df = df.reset_index(drop=False)
        df = df.drop(columns=['latitude', 'longitude','crs','height','station'])
        df.time = pd.to_datetime(df['time'])
        df = df.set_index(['time'])
        df['station_name'] = df.station_name.str.decode("utf-8")
        df = df[df.station_name == site]

        prec = 'PREC_sur'

        precip_avg = df[prec].resample('M').mean()
        plt.plot(precip_avg, label=os.path.basename(data), c=col)
        plt.ylabel('Precip Average (kg m-2 s-1)')
        plt.xlabel('Month of Yr')
        mer.close()
    
    f = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/scaled/to_compare/special/interpolated.nc', engine="netcdf4")
    mdf = f.to_dataframe()
    # LAT: [62.5, 63.5, 64.7] --> [5, 7, 9]   LON: [-114.375, -108.75, -110.625] --> [42, 51, 48]
    mdf = mdf.PRECTOTCORR[:,9,:]
    mdf = mdf.resample("M").mean()
    f.close()
    plt.plot(mdf, label='Interpolated MERRA')
    
    f = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/scaled/to_compare/special/downloaded.nc', engine="netcdf4")
    mdf = f.to_dataframe()
    # LAT: [62.5, 63.5, 64.7] --> [5, 7, 9]   LON: [-114.375, -108.75, -110.625] --> [42, 51, 48]
    mdf = mdf.drop(['latitude', 'longitude'], axis=1)
    mdf = pd.concat([mdf.loc[:, 5, 42].PRECTOTCORR.rename("yk")],axis=1)
    mdf = mdf.resample("M").mean()
    mdf = mdf[mdf.index < pd.to_datetime('2020-01-01')]
    f.close()
    plt.plot(mdf, label='2D Download MERRA')

    plt.xticks(rotation=70)

    plt.legend()
    plt.title(site)
    plt.savefig(f'/home/hma000/accomatic-web/tests/plots/merra_test/transect/{site}.png')
    plt.clf()
    plt.close()
    print(site)





