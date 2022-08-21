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

plot = True
if plot:
    df = pd.read_csv('terrains_df.csv')
    df.time = pd.to_datetime(df['time'], format="%Y-%m-%d")
    df = df[df.time.dt.month > 5]
    df = df[df.time.dt.month < 9]
    df = df.reset_index()
    df = df.set_index(['time', 'simulation'])
    # Setting custom colour palette for seaborn plots
    palette = ["#1CE1CE", "#008080", "#F3700E", "#F50B00", "#59473C"]
    sns.set_palette(sns.color_palette(palette))
    sns.set_context('poster')
    sns.despine()
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(x='time', y='soil_temperature', linewidth = 4, data = df, ax=ax, hue='simulation')

    # Set title and labels for axes
    ax.set(xlabel="Date",
        ylabel="GST (C˚)")
    
    ax.legend(['Clay','Peat','Sand','Rock', 'Gravel'],loc='best')
    
    # Define the date format
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)
    
    plt.savefig("terrains_thick.png", dpi=300, legend=False, transparent=True)
    

trial = False
if trial:

    f = xr.open_dataset('/fs/yedoma/usr-storage/hma000/KDI/KDI_obs.nc')
    month_length = f.time.dt.days_in_month

    weights = (
        month_length.groupby("time.season") / month_length.groupby("time.season").sum()
    )

    # Test that the sum of the weights for each season is 1.0
    np.testing.assert_allclose(weights.groupby("time.season").sum().values, np.ones(4))

    # Calculate the weighted average
    f = f['soil_temperature']
    ds_weighted = (f * weights).groupby("time.season").sum(dim="time")
    # ds_unweighted = f.groupby("time.season").mean("time")

    sns.lineplot(data = ds_weighted)
    plt.savefig("weighted.png")