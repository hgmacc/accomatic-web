import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import sys
from matplotlib.dates import DateFormatter

TYPE = 'obs'

# Loading in data
FILE = {'mod': '/home/hma000/accomatic-web/tests/test_data/KDI_10CM_23Sites.nc',
        'obs': '/fs/yedoma/usr-storage/hma000/KDI/KDI_obs.nc'}
        
def getdf():
    if TYPE == 'obs':
        f = xr.open_dataset(FILE['obs'])
        df = f.soil_temperature.to_dataframe()
    
    elif TYPE == 'mod':
        # Get dataset
        f = xr.open_dataset(FILE[TYPE], group='geotop')
        df = f.to_dataframe()
        
        # Drop dumb columns and rename things
        df = df.drop(['model', 'pointid'], axis=1).rename({'Date': 'time', 'Tg': 'soil_temperature'}, axis=1) 
        df = df.reset_index(level=(0,1), drop=True)
        df = df.reset_index(drop=False)

        # Merge simulation and sitename colummn 
        df.simulation = df.sitename  + ',' + df.simulation 

        # Setting up time index
        df.time = pd.to_datetime(df['time'])
        df.index = df.time
        df = df.drop(['time', 'sitename'], axis=1)

    else:
        print("Need to specify 'obs' or 'mod' type.")
        sys.exit()

    return df

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