import glob
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys

try: f.close()
except: pass


downloaded_pth = '/home/hma000/storage/merra_precip_test/ykl/'
all_ncs = glob.glob(downloaded_pth + '/*/ts_*.nc')
sites_pth = '/home/hma000/storage/yk_kdi_ldg/par/ykl.csv'
VAR_DICT = {'era5':'tp','merra2':'PRECTOTCORR', 'jra55':'Total precipitation'}


def get_sites(sites_pth, opt='df'):
    """
    Auto formats sites csv for plotting.

    Args:
        sites_pth (str): Path to csv file
        opt (str): Either "df" or "dic", depending on what return format you want

    Returns:
        _type_: Either "df" or "dic" of sites
    """
    sites = pd.read_csv(sites_pth, usecols=['station_number','station_name','longitude_dd','latitude_dd'])
    sites.longitude_dd = round(sites.longitude_dd * 4) / 4
    sites.latitude_dd = round(sites.latitude_dd * 4) / 4
    sites = sites.drop_duplicates('longitude_dd').head()
    if opt == 'dic': 
        sites = dict(zip(sites.station_number, sites.station_name))
    return sites

def get_nc_point_data(pth, site_dict):
    # Getting precip variable name from dict
    #if os.path.basename(pth).split('_')[2] == "scaled.nc": var = 'PREC_sur'
    #else: var = VAR_DICT[os.path.basename(pth).split('_')[1]]
    var = 'PREC_sur'
    # NC data to df
    f = xr.open_mfdataset(pth, engine="netcdf4")
    df = f.to_dataframe()
    
    # Drop useless columns & format
    df = df.reset_index(drop=False)
    df = df[['time','station', var]]
    df.time = pd.to_datetime(df['time'])
    df = df[df.station.isin(site_dict.keys())]
    df.station = [site_dict[i] for i in df.station]
    df = df.set_index(['time', 'station'])
    df_list = [df.loc[:, stn, :][var].rename(stn) for stn in site_dict.values()]
    df = pd.concat(df_list, axis=1).resample("W-MON").mean()
    return df

def swap_all_merr_values():    
    mer = xr.open_mfdataset('/home/hma000/storage/yk_kdi_ldg/scaled/ykl/MERRA2.nc', engine="netcdf4")
    mer_fixed = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/yk_merra_bandaid/scaled/scaled_merra2_1h_scf1.0.nc', engine="netcdf4")
    for var in ['PRESS_pl', 'AIRT_pl', 'AIRT_sur', 'PREC_sur', 'RH_sur', 'WSPD_sur', 'WDIR_sur', 'SW_sur', 'LW_sur', 'SH_sur']:
        n = mer_fixed[var][:].values
        n = np.repeat([n], 14, axis=0).T[0,:,:]
        #print(mer[var][:,:14].values.shape)
        #print(n.shape); sys.exit()
        mer[var][:,:14] = n
        print(f"{var} has been processed.")
    mer.to_netcdf(path='/home/hma000/storage/merra_precip_test/yk_merra_bandaid/MERRA2_fixed.nc', mode='w')

def swap_only_merr_precip_values():    
    mer_fixed = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/yk_merra_bandaid/MERRA2_fixed.nc', engine="netcdf4")
    mer = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/ykl_ncs/ts_merra2_scaled.nc', engine="netcdf4")

    for var in ['PREC_sur']:
        n = mer_fixed[var][:].values
        n = np.repeat([n], 14, axis=0).T[0,:,:]
        #print(mer[var][:,:14].values.shape)
        #print(n.shape); sys.exit()
        mer[var][:,:14] = n
        print(f"{var} has been processed.")
    mer.to_netcdf(path='/home/hma000/storage/merra_precip_test/temp/scaled_merra2_1h_scf3.0.nc', mode='w')

swap_only_merr_precip_values()

def get_mer_latlon(mdf, l):
    # Get lat lon indeces for merra data where l = [lat, lon]
    lat = mdf.iloc[(mdf['latitude']-l[0]).abs().argsort()[:1]].lat.values[0]
    lon = mdf.iloc[(mdf['longitude']-l[1]).abs().argsort()[:1]].lon.values[0]
    
    return [lat, lon]
        
def read_2D_merra(sites):
    f = xr.open_mfdataset(downloaded_pth + 'merra2_downloaded.nc', engine="netcdf4")
    mdf = f.to_dataframe()
    coords = mdf[['latitude', 'longitude']].droplevel(0).reset_index(drop=False)
    mdf = mdf.drop(['latitude', 'longitude'], axis=1)
    # Get all lat lon row/col indeces from 2D merra data
    latlon = [get_mer_latlon(coords, [lat, lon]) for lat, lon in zip(sites.latitude_dd, sites.longitude_dd)]
    # Pull out PRECTOTCORR timeseries for each grid cell & concatonate
    df_list = [mdf.loc[:, loc[0], loc[1]].PRECTOTCORR.rename(sitename) for loc, sitename in zip(latlon, sites.station_name)]
    mdf = pd.concat(df_list, axis=1).resample("W-MON").mean()
    f.close()
    return mdf

def read_2D_era5(sites):
    f = xr.open_mfdataset(downloaded_pth + 'era5_downloaded.nc', engine="netcdf4")
    lon, lat = xr.DataArray(sites.longitude_dd.tolist(), dims="points"), xr.DataArray(sites.latitude_dd.tolist(), dims="points")
    edf = f['tp'].sel(longitude=lon, latitude=lat, method="nearest")
    edf = edf.to_dataframe().drop(['longitude', 'latitude'], axis=1)
    df_list = [edf.loc[:,loc,:].tp.rename(sitename) for loc, sitename in zip(range(len(sites)), sites.station_name)]
    edf = pd.concat(df_list, axis=1).resample("W-MON").mean()
    f.close()
    return edf

def read_2D_jra55(sites):
    f = xr.open_mfdataset(downloaded_pth + 'jra55_downloaded.nc', engine="netcdf4")
    lon, lat = xr.DataArray((sites.longitude_dd + 341.075).tolist(), dims="points"), xr.DataArray(sites.latitude_dd.tolist(), dims="points")
    jdf = f['Total precipitation'].sel(longitude=lon, latitude=lat, method="nearest")
    jdf = jdf.to_dataframe().drop(['longitude', 'latitude'], axis=1)
    
    df_list = [jdf.loc[:,loc,:]["Total precipitation"].rename(sitename) for loc, sitename in zip(range(len(sites)), sites.station_name)]    
    jdf = pd.concat(df_list, axis=1).resample("W-MON").mean() / 86400
    f.close()
    return jdf

