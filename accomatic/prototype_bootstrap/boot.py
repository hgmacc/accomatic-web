import pandas as pd
import sys
import random
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np

from accomatic.NcReader import *
from accomatic.Experiment import *
from accomatic.Stats import *
from accomatic.Plotting import *

import warnings
warnings.simplefilter("ignore") 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"


def get_data(site) -> pd.DataFrame:
    e = Experiment('/home/hma000/accomatic-web/tests/test_data/toml/NOV_NWT.toml')
    df = e.mod(site).join(e.obs(site)).dropna()
    df['ens'] = df[['era5','merra2','jra55']].mean(axis=1)
    return df

def remove_days(df, chunk_size, reps, reindex=True) -> pd.DataFrame:
    """
    chunk_size (int): Number of consecutive days to remove
    reps (int): How many chunks to remove
    """
    if chunk_size == 0 or reps == 0:
        return df
    for i in range(reps):
        nrows = range(df.shape[0])
        ix = random.randint(nrows.start, nrows.stop-chunk_size)
        df = df.drop(df.iloc[ix:ix+chunk_size, :].index)
        # Re-index to fill any missing days
    if reindex:
        df = df.reindex(pd.date_range(df.index.min(), df.index.max()))
    return df

def plot_ts_missing_days(df, site) -> None:
    models = df.drop(columns='obs').columns
    fig, ax = plt.subplots(figsize=(15, 6))
    plt.rcParams["font.size"] = "16"
    plt.plot(df.index, df['obs'], 'k', label='obs')
    for mod in models:
        plt.plot(df.index, df[mod], ':', label=mod)

    # Set title and labels for axes
    ax.set(xlabel="Date", ylabel="GST (C)")

    # Highlighting missing data
    for i in df[pd.isnull(df.obs)].index:
        ax.axvspan(i, i+timedelta(days=1), facecolor='green', edgecolor='none', alpha=.5)

    plt.ylim((-25,18))
    plt.title(site)
    plt.legend()
    plt.xticks(rotation = 18)
    plt.savefig(f"/home/hma000/accomatic-web/accomatic/prototype_bootstrap/plots/ts_{site}.png")
    
# As chunk_size increases, how do bootstrap results change?
# As reps increases, how do bootstrap results change?

# Write bootstrap function :) 

def boot(df, sim, acco, boot_size=1000, consecutive_days_slice=10):
    nrows = range(df.shape[0])
    res = []
    for i in range(boot_size):
        # Select n consecutive days 
        ix = random.randint(nrows.start, nrows.stop-consecutive_days_slice)
        a = df.iloc[ix:ix+consecutive_days_slice, :]
        res.append(acco_measures[acco](a.obs, a[sim]))

    res = np.array(sorted(res)[50:950])
    res = res[(res<10) & (res>-10)]

    return res


def boot_plot(data, site, stat, labels):
    fig, ax = plt.subplots(figsize=(len(data)+2, 10))
    bp = ax.boxplot(data, whis=1.5, patch_artist=True)
    ax.set_title(f"{stat} at {site}.")
    for patch, color in zip(bp['boxes'], get_color_gradient("#036c5f", "#b3e0dc", len(data))):
        patch.set_facecolor(color)
    for median in bp['medians']:
        median.set_color('#59473C')
    ax.set_xticklabels(labels)
    ax.set_xlabel(TITLE)
    plt.savefig(f'/home/hma000/accomatic-web/accomatic/prototype_bootstrap/plots/bs_{OPT}_{stat}_{site}_rep.png')
    plt.clf()
    

def bs_threshold_exp(site, stat):
    df = get_data(site)[['ens', 'obs']]
    rep_list = [0, 25, 50, 75, 100]
    if OPT == 'chunk':
        data = [boot(remove_days(df, chunk_size=i, reps=1, reindex=False), 'ens', stat) for i in rep_list]
    if OPT == 'reps':
        data = [boot(remove_days(df, chunk_size=1, reps=i, reindex=False), 'ens', stat) for i in rep_list]        
    boot_plot(data, site, stat, rep_list)
    
#OPT = 'chunk' # 'reps' or 'chunk'
#TITLE = 'n consective days removed once.' 
OPT = 'reps' 
TITLE = 'one single day removed n times'
#bs_threshold_exp(site='NGO-DD-2035', stat='MAE')#, title='Consecutive number of days removed ONCE.')

plot_ts_missing_days(remove_days(get_data('NGO-DD-2035')[['ens', 'obs']], chunk_size=100, reps=1, reindex=True), 'NGO-DD-2035')