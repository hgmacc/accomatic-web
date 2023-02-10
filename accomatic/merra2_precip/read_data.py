import glob
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

try: f.close()
except: pass


all_ncs = glob.glob('/home/hma000/storage/merra_precip_test/ykl/ts_*.nc')

sites = pd.read_csv('/home/hma000/storage/yk_kdi_ldg/par/ykl.csv', usecols=['station_name','longitude_dd','latitude_dd'])
sites.longitude_dd = round(sites.longitude_dd * 4) / 4
sites.latitude_dd = round(sites.latitude_dd * 4) / 4
sites = sites.drop_duplicates('longitude_dd').head()

def get_latlon(mdf, l):
    # Get lat lon indeces for merra data where l = [lat, lon]
    lat = mdf.iloc[(mdf['latitude']-l[0]).abs().argsort()[:1]].lat.values[0]
    lon = mdf.iloc[(mdf['longitude']-l[1]).abs().argsort()[:1]].lon.values[0]
    
    return [lat, lon]

downloaded_pth = '/home/hma000/storage/merra_precip_test/ykl/'
        
def read_merra(sites):
    f = xr.open_mfdataset(downloaded_pth + 'merra2_downloaded.nc', engine="netcdf4")
    mdf = f.to_dataframe()
    coords = mdf[['latitude', 'longitude']].droplevel(0).reset_index(drop=False)
    mdf = mdf.drop(['latitude', 'longitude'], axis=1)
    # Get all lat lon row/col indeces from 2D merra data
    latlon = [get_latlon(coords, [lat, lon]) for lat, lon in zip(sites.latitude_dd, sites.longitude_dd)]
    # Pull out PRECTOTCORR timeseries for each grid cell & concatonate
    df_list = [mdf.loc[:, loc[0], loc[1]].PRECTOTCORR.rename(sitename) for loc, sitename in zip(latlon, sites.station_name)]
    mdf = pd.concat(df_list, axis=1)
    mdf = mdf.resample("W-MON").mean()
    f.close()
    return mdf


def read_era5(sites):
    f = xr.open_mfdataset(downloaded_pth + 'era5_downloaded.nc', engine="netcdf4")
    lon, lat = xr.DataArray(sites.longitude_dd.tolist(), dims="points"), xr.DataArray(sites.longitude_dd.tolist(), dims="points")
    edf = f['tp'].sel(longitude=lon, latitude=lat, method="nearest")
    edf = edf.to_dataframe().drop(['longitude', 'latitude'], axis=1)
    df_list = [edf.loc[:,loc,:].tp.rename(sitename) for loc, sitename in zip(range(len(sites)), sites.station_name)]
    edf = pd.concat(df_list, axis=1)
    edf = edf.resample("W-MON").mean()
    f.close()
    return edf


    
def read_jra55(sites):
    f = xr.open_mfdataset(downloaded_pth + 'jra55_downloaded.nc', engine="netcdf4")
    lon, lat = xr.DataArray([226.7, 232, 230.5], dims="points"), xr.DataArray([62.5, 63.5, 64.7], dims="points")
    jdf = f['Total precipitation'].sel(longitude=lon, latitude=lat, method="nearest")
    jdf = jdf.to_dataframe().drop(['longitude', 'latitude'], axis=1)
    jdf = pd.concat([jdf.loc[:,0,:]["Total precipitation"].rename("yk"), 
            jdf.loc[:,1,:]["Total precipitation"].rename("kdi"),
            jdf.loc[:,2,:]["Total precipitation"].rename("ldg")], axis=1)
    jdf = jdf.resample("W-MON").mean() / 86400
    f.close()
    return jdf

a = read_jra55(sites)
print(a.head())