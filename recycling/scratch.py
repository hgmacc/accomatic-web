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


def nse(obs, mod):
    return 1-(np.sum((mod-obs)**2)/np.sum((obs-np.mean(obs))**2))

# this could be super wrong
def willmot_d(obs, mod):
    return np.mean((1-(((obs-models[mod])**2)/(abs(models[mod]-np.mean(obs))+abs(obs-np.mean(obs))**2))))

def bias(obs, mod):
    return np.mean(mod-obs)

def rmse(obs, mod):
    return(mean_squared_error(obs, mod, squared=False))

stats = {"RMSE" : rmse, 
        "R2" : mean_absolute_error, 
        "MAE" : r2_score,
        "NSE" : nse,
        "WILL" : willmot_d,
        "BIAS" : bias}
  
# printing original dictionary
res = stats['RMSE'](obs, mod)  # -> str(res) = '0.97'

# for model in models       
    # for acco in acco_list:
        stats_dict[mod+'-'+szn] = [stat[](mod, obs), stat[](mod, obs), stat[](mod, obs)]
result []
for s in acco_list:
    result.append(float(stats[s](obs, mod))
stats_dict[mod+szn] = result


def generate_stats(df, szn, acco_list):
    # Set up x and y data for analysis
    obs = df.soil_temperature
    models = df.drop(['soil_temperature'], axis=1)

    stats_dict = {}

    for mod in models:
        stats_dict[mod+szn] = map(func, acco_list)
        
        result []

        for s in acco_list:
            result.append(float(stats[s](obs, mod))
        
        stats_dict[mod+szn] = result

    return stats_dict