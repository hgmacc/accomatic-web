import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from matplotlib.dates import DateFormatter

        
def getdf():
        # Get dataset
        o = xr.open_dataset(file['obs'])
        odf = o.to_dataframe()
        
        # Clean up columns
        odf = odf.drop(['latitude', 'longitude', 'elevation', 'depth'], axis=1).rename({'platform_id': 'sitename'}, axis=1) 

        # Fix index
        odf = odf.reset_index(level=(1), drop=True)
        odf.sitename = [line.decode("utf-8") for line in odf.sitename]
        odf = odf.set_index(odf.sitename, append=True)
        odf = odf.drop(['sitename'], axis=1)
        
        # Get dataset
        m = xr.open_dataset(file["mod"], group='geotop')
        mdf = m.to_dataframe()
        
        # Drop dumb columns and rename things
        mdf = mdf.drop(['model', 'pointid'], axis=1).rename({'Date': 'time', 'Tg': 'soil_temperature'}, axis=1) 
        mdf = mdf.reset_index(level=(0,1), drop=True)
        mdf = mdf.reset_index(drop=False)

        # Merge simulation and sitename colummn 
        mdf.simulation = mdf.sitename  + ',' + mdf.simulation 

        # Setting up time index
        mdf.time = pd.to_datetime(mdf['time'])
        mdf.index = df.time
        df = mdf.drop(['time', 'sitename'], axis=1)




plot = True
if plot:
    df = getdf()
    
    # Setting custom colour palette for seaborn plots
    palette = ["#1CE1CE", "#008080", "#F3700E", "#F50B00", "#59473C"]
    sns.set_palette(sns.color_palette(palette))
    sns.set_context('poster')
    sns.despine()
    
    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(48, 12))

    # Add x-axis and y-axis
    #sns.lineplot(x='time', y='soil_temperature', data = df, ax=ax)
    sns.lineplot(x='time', y='soil_temperature', data = df, ax=ax)

    # Set title and labels for axes
    ax.set(xlabel="Date",
        ylabel="GST (C˚)",
        title="Ground Surface Temperatures (10 cm) at KDI n = 23")

    # Define the date format
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)
    
    plt.rcParams["font.family"] = "Times New Roman"
    plt.savefig(TYPE + ".png", dpi=100, transparent=True) 
    


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