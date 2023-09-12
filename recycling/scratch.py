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


"""
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


"""
def generate_stats(df, szn, acco_list):
    # Set up x and y data for analysis
    obs = df.soil_temperature
    models = df.drop(['soil_temperature'], axis=1)

    stats_dict = {}

    for mod in models:
        stats_dict[mod+szn] = map(func, acco_list)
        
        result = []

        for s in acco_list:
            result.append(float(stats[s](obs, mod)))
        
        stats_dict[mod+szn] = result

    return stats_dict





rank_comparison_dict = {}

# Bootstrap-t steps

# 1. 
# 1. x_1, X_2, ...., x_n is the sample drawn from distribution F
    # x_n is the accordance value at point _n
# 2. u is the mean computed from the sample, so mean(x_1, x_2,...,x_n)
# 3. F* is the resampling distribution
# 4. x*_1, x*_2,...,x*_n is a resample of the same size as the original sample
# 5. u* is the mean of the resample
# Bootstrap principle states that F* approximates F
# Variation of u is well-approximated by variation of u*

# Confidence interval degree is 1 - ci_value %
ci_value=0.05
resample_size = 0     # if 0, resample size will be same as sample
resample_freq = 10000

#rank_comparison_dict = {}

# positions in the resample lists that represent the upper and lower ci bounds 
lower_ci = int(round(resample_freq * ci_value/2)) - 1
upper_ci = int(round(resample_freq * (1-ci_value/2))) - 1

for m in all_accordance_funcs:

    rank_comparison_dict[m] = pd.DataFrame()

    rank_comparison_dict[m]['mean'] = model_means[m]
    rank_comparison_dict[m]['rank'] = model_ranks[m]


    # Accordance values for all points
    pop_sample = all_accordance_results[m]
    
    # place to store generated CI values
    ci_min_series = pd.Series()
    ci_max_series = pd.Series()
    
    # all values by model
    for col in pop_sample:
        
        
        current_sample = pop_sample[col]
        current_sample_mean = model_means[m].loc[col]
        current_sample_sem = stats.sem(current_sample)
        # our values for the 10000 samples
        resample_deltas = []
        
        # repeat resample_freq times
        for i in range(0,resample_freq):
            
            num_elements = resample_size
            
            if (resample_size == 0):
                num_elements = len(current_sample)
            
            
            # Calculate t-value for each bootstrap sample
            # formula is t = (sample_mean - population mean) / (sample sd / sqrt(sample size))
            # population mean is normalized to 0 for process
            
            
            # take mean of resample with replacement of num_elements elements, subtract from sample mean
            # Create bootstrap resample
            bootstrap_sample = random.choices(current_sample, k=num_elements)
            
            # calculate t-statistic for resample
            t_stat = (mean(bootstrap_sample) - current_sample_mean)/stats.sem(bootstrap_sample)
            #t_stat = (mean(bootstrap_sample) - current_sample_mean)/(stdev(bootstrap_sample)/math.sqrt(len(boostrap_sample))
            
            #add t-statistic to list of t-stats for CI generation
            resample_deltas.append(t_stat)
            
            #resample_deltas.append((mean(random.choices(current_sample, k=num_elements)) - current_sample_mean)/SE)
            
        # Sort differences
        resample_deltas.sort()
        
        # Set lower and upper limits of confidence interval for this model
        
        ci_max_series[col] = current_sample_mean - resample_deltas[lower_ci]*current_sample_sem
        ci_min_series[col] = current_sample_mean - resample_deltas[upper_ci]*current_sample_sem
    
    # Set the CI min and max values for this measure on all models
    rank_comparison_dict[m]['ci_min'] = ci_min_series 
    rank_comparison_dict[m]['ci_max'] = ci_max_series



