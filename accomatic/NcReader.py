import os
import re
import sys
import typing
from datetime import datetime, timedelta
from os import path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xarray as xr



def create_acco_nc(exp) -> None:
    acco = xr.Dataset(
        str(exp.model_pth + "_acco_results.nc"), mode="w", format="NETCDF4"
    )
    # build stats
    # populate file

    acco.close()


def read_geotop(file_path="", sitename="", ens=False, depth=False) -> pd.DataFrame:
    # Get dataset
    m = xr.open_dataset(file_path, group="geotop")

    m["sitename"] = m.sitename.str.replace(r"_site", "")

    if sitename != "" and type(sitename) != list:  # sitename provided
        m = m.where(m.sitename == sitename, drop=True)
    elif type(sitename) == list:  # sitename list provided
        m = m.where(m.sitename.isin(sitename), drop=True)
        
    mdf = m.to_dataframe()
    
    # Drop dumb columns and rename things
    mdf = mdf.drop(["model", "pointid"], axis=1).rename(
        {"Date": "time", "Tg": "soil_temperature"}, axis=1
    )
        
    mdf = mdf.reset_index(level=("time"), drop=True).reset_index(drop=False)

    
    if depth: 
        print(f"Model clipped to {depth}m depth.")
        mdf = mdf[mdf.soil_depth.round(1) == depth]
    
    mdf = mdf.drop(["soil_depth"], axis=1)

    # Fix simulation colummn
    mdf.simulation = [
        line.split('_')[2] for line in mdf.simulation
    ]  # (...)led_merr_3e66cca -> merra2

    # Setting up time index
    mdf.time = pd.to_datetime(mdf["time"]).dt.date
    mdf = mdf.drop_duplicates(subset=['simulation', 'time', 'sitename'])
    mdf = mdf.set_index([mdf.time, mdf.sitename, mdf.simulation], append=True)
    
    mdf = mdf.drop(["time", "sitename", "simulation"], axis=1)
    mdf = mdf.reset_index(level=(0), drop=True)    
    mdf = mdf.dropna()
    mdf = mdf.unstack(level=2).soil_temperature
    mdf['ens'] = mdf[['era5', 'jra55', 'merra2']].mean(axis=1)

    return mdf


def average_obs_site(odf) -> pd.DataFrame:
    """
    Averaging GST output for each site.
    """
    odf["sitename"] = odf.index.get_level_values("sitename").str[:]
    # KDI-E-Org_02 -> KDI-E-Org
    odf.sitename = [line.split('_')[0] for line in odf.sitename]  
    # ROCK1A -> ROCK1
    odf.sitename = [line.rstrip('ABC') for line in odf.sitename]

    # Drop sitename index so we can use new 'sitename' col to avg over non-unique sitenames
    odf = odf.reset_index(level=(1), drop=True)

    # Average 'soil_temperature' over 'sitename'
    odf["temp_site_date"] = odf.sitename + odf.index.get_level_values("time").astype(
        str
    )
    odf.soil_temperature = odf.groupby("temp_site_date")["soil_temperature"].transform(
        "mean"
    )

    odf = odf.drop_duplicates(subset=["temp_site_date"], keep="first")

    # Cleaning up so df format is still (time, sitename) : soil_temperature
    odf = odf.set_index(odf.sitename, append=True)
    odf = odf.drop(["temp_site_date", "sitename"], axis=1)
    odf = odf.rename(columns={"soil_temperature": "obs"})
    
    return odf


def read_nc(file_path, sitename = "", avg=True, depth=False) -> pd.DataFrame:
    # Get dataset
    o = xr.open_dataset(file_path)
    odf = o.to_dataframe()

    # Clean up columns
    odf = odf.drop(["latitude", "longitude", "elevation"], axis=1).rename(
        {"platform_id": "sitename"}, axis=1
    )
    
    # Fix index
    odf = odf.reset_index(level=(1), drop=True)
    odf.sitename = [line.decode("utf-8") for line in odf.sitename]
    odf = odf.set_index(odf.sitename, append=True)
    odf = odf.drop(["sitename"], axis=1)
        
    # if not assuming GST, round depth to 0.1 / 0.5 / 1.0
    if depth: odf = odf[odf.depth.round(1) == float(depth)]    
    odf = odf.drop(["depth"], axis=1)
    
    # avg toggle used to average gst observations where > 1 logger
    if avg: odf = average_obs_site(odf)
    
    # If missing_data_exp for bootstrap: remove data
    missing_data_exp = False
    if missing_data_exp:
        list_of_dates = odf.index.get_level_values(0).unique().tolist()
        percent = 0.25
        print(percent)
        from numpy.random import default_rng
        rng = default_rng()
        indices = rng.choice(len(list_of_dates), 
                        size=int(np.round(percent * len(list_of_dates))), 
                        replace=False)
        
        list_of_dates = [list_of_dates[i] for i in indices]
        odf.drop(list_of_dates, axis=0, inplace=True)
    
    # Selecting only sites specified in toml file    
    odf = odf[odf.index.get_level_values(1).isin(sitename)]

    odf = odf.dropna()
    print(f"Observations: {len(odf.index.get_level_values(1).unique())} sites at {depth}m depth.")

    return odf


