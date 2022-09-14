import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from matplotlib.dates import DateFormatter


def get_single_df():
    filepth = '/home/hma000/accomatic-web/tests/test_data/terrain_output.nc'
    # Get dataset
    m = xr.open_dataset(filepth, group='geotop')
    mdf = m.to_dataframe()
    
    # Drop dumb columns and rename things
    mdf = mdf.drop(['model', 'pointid'], axis=1).rename({'Date': 'time', 'Tg': 'soil_temperature'}, axis=1) 
    mdf = mdf.reset_index(level=(0,1), drop=True)
    mdf = mdf.reset_index(drop=False)

    # Merge simulation and sitename colummn 
    mdf.simulation = mdf.sitename  + ',' + mdf.simulation 
    mdf.sitename = [line.strip("_site") for line in mdf.sitename]

    # Setting up time index
    mdf.time = pd.to_datetime(mdf['time']).dt.date
    mdf = mdf.set_index([mdf.time, mdf.sitename], append=True)
    mdf = mdf.drop(['time', 'sitename'], axis=1)

    return mdf

csv = False
if csv:
    filepth = '/home/hma000/accomatic-web/tests/test_data/terrain_output.nc'
    # Get dataset
    m = xr.open_dataset(filepth, group='geotop')
    mdf = m.to_dataframe()
    mdf = mdf.loc[mdf.sitename.str.contains('KDI-E-Org2_site'),:]

    # Drop dumb columns and rename things
    mdf = mdf.drop(['model', 'pointid'], axis=1).rename({'Date': 'time', 'Tg': 'soil_temperature'}, axis=1) 
    mdf = mdf.reset_index(level=(0,1), drop=True)
    mdf = mdf.reset_index(drop=False)

    # Merge simulation and sitename colummn 
    mdf.simulation = mdf.simulation.str[-7:]

    # Setting up time index
    mdf.time = pd.to_datetime(mdf['time'], format="%Y-%m-%d")
    mdf = mdf[mdf.time.dt.month <= 9]
    mdf = mdf.set_index([mdf.time, mdf.simulation], append=True)
    df = mdf.drop(['time', 'sitename', 'simulation'], axis=1)
    df.to_csv("terrains_df.csv")

