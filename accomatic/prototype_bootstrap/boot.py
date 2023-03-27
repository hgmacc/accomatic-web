import pandas as pd
import sys
import random
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
import numpy as np
import os

from accomatic.NcReader import *
from accomatic.Experiment import *
from accomatic.Stats import *
from accomatic.Plotting import *

import warnings
warnings.simplefilter("ignore") 

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"


def get_data(e, site) -> pd.DataFrame:
    if site == 'all sites':
        # Average over all sites
        df = e.mod().join(e.obs()).dropna()
        df = df.groupby('time').mean()
    elif site == 'terr':
        # Return all sites
        df = e.mod().join(e.obs()).dropna()
    else:
        # Return only one site
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


def ten_day_boot(df, sim, acco, boot_size=1000, consecutive_days_slice=10):
    nrows = range(df.shape[0])
    res = []
    for i in range(boot_size):
        # Select n consecutive days 
        ix = random.randint(nrows.start, nrows.stop-consecutive_days_slice)
        a = df.iloc[ix:ix+consecutive_days_slice, :]
        res.append(acco_measures[acco](a.obs, a[sim]))

    res = np.array(sorted(res)[50:950])
    res = res[(res<10) & (res>-10)]
    print('One done')
    return res


def crazy10_day_boot(df, sim='ens', acco='MAE', boot_size=1000, chunk_size=1, reps=1):
    res = []
    df = df[['obs', sim]]    
    for i in range(100):
        smol_df = remove_days(df, chunk_size=chunk_size, reps=reps)
        nrows = range(df.shape[0])
        for j in range(1000):
            a = get_10_days(smol_df, nrows)
            res.append(acco_measures[acco](a.obs, a[sim]))

    res = np.array(sorted(res)[50:950])
    res = res[(res<10) & (res>-10)]
    return res


def get_10_days(df, nrows):
    consecutive_days_slice=10
    ix = random.randint(nrows.start, nrows.stop-(consecutive_days_slice+1))
    a = df.iloc[ix:ix+consecutive_days_slice, :]
    counter = 0
    while a.ens.isna().sum() > 0 and counter !=10:
        ix = random.randint(nrows.start, nrows.stop-(consecutive_days_slice+1))
        a = df.iloc[ix:ix+consecutive_days_slice, :]
        counter = counter + 1
    if counter == 10:
        print('yikes'); sys.exit()
    return a
        
def simple_10_day_boot(df, sim='ens', acco='MAE', boot_size=1000, chunk_size=1, reps=1):
    res = []
    df = df[['obs', sim]]
    df = remove_days(df, chunk_size=chunk_size, reps=reps)
    nrows = range(df.shape[0])
    
    for i in range(boot_size):
        a = get_10_days(df, nrows)
        res.append(acco_measures[acco](a.obs, a[sim]))

    res = np.array(sorted(res)[50:950])
    res = res[(res<10) & (res>-10)]
    return res

def simple_boot(df, sim='ens', acco='MAE', boot_size=1000, chunk_size=1, reps=1):
    res = []
    df = df[['obs', sim]]
    for i in range(boot_size):
        a = remove_days(df, chunk_size=chunk_size, reps=reps, reindex=False)
        res.append(acco_measures[acco](a.obs, a[sim]))
    res = np.array(sorted(res)[50:950])
    res = res[(res<10) & (res>-10)]
    return res

def boot_boxplot(data, site, stat, labels):
    fig, ax = plt.subplots(figsize=(len(data)+2, 10))
    bp = ax.boxplot(data, whis=1, 
                    showbox=False, 
                    showfliers=False, 
                    patch_artist=True, 
                    meanline=True, 
                    medianprops=dict(linewidth=0, linestyle=None), 
                    meanprops= dict(linestyle='-', linewidth=2.5, color='black'), 
                    boxprops=dict(linewidth=1),
                    showmeans=True)
    """
    for patch, color in zip(bp['boxes'], get_color_gradient("#036c5f", "#b3e0dc", len(data))):
        patch.set_facecolor(color)
    
    for median in bp['medians']:
        median.set_color('#59473C')
    """    
    ax.set_title(f"{OPT}: {stat} at {site}.")
    ax.set_xticklabels(labels)
    ax.set_xlabel(TITLES[OPT])
    ax.set_ylim(bottom=1)
    plt.savefig(f'/home/hma000/accomatic-web/accomatic/prototype_bootstrap/plots/box_{OPT}_{stat}_{site}_rep.png')
    plt.clf()

def boot_vioplot(data, site, stat, sim, label, title=''):
    fig, ax = plt.subplots(figsize=(len(data)+4, 12))
    bp = ax.violinplot(data, showmeans=True)

    for patch, color in zip(bp['bodies'], get_color_gradient("#b3e0dc", "#036c5f", len(label))):
        patch.set_facecolor(color) 
        patch.set_alpha(1.0)
        
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = bp[partname]
        vp.set_edgecolor('#000000')
        vp.set_linewidth(1)
     
    # This is the most annoying line of code I've ever written. 
    # ax.set_xticks([i for i in range(0, len(label), 2)], labels=[str(i) for i in label[::2]])

    ax.set_title(f"{sim} at {site}")
    ax.set_xlabel(TITLES[OPT])
    ax.set_ylabel(stat)
    ax.set_ylim(bottom=0, top=2.5)
    
    ax.set_xticks(range(1, len(label)+1), labels=label, rotation=70)

    if title == '':
        title = f"vio_{OPT}_{stat}_{site}_rep"
    plt.savefig(f'{os.path.dirname(os.path.realpath(__file__))}/{title}.png')
    plt.clf()

def bs_threshold_exp(stat, site='all sites', sim='ens'): 
    df = get_data(EXP, site)[[sim, 'obs']]    
    rep_list = [i for i in range(0, 1000, 50)]
    if OPT == 'c':
        data = [simple_boot(df, sim, stat, chunk_size=i) for i in rep_list]
    if OPT == 'r':
        data = [simple_boot(df, sim, stat, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, rep_list)
    
def bs_threshold_terr_exp(stat, terr, sim='ens'): 
    df = get_data(EXP,'terr')[[sim, 'obs']]    
    terr_list = []
    for i in df.index.get_level_values(1):
        try: terr_list.append(EXP.terr_dict()[i])
        except KeyError:
            terr_list.append(-1)
    df['terrain'] = terr_list
    df = df[df.terrain == str(terr)].drop(["terrain"], axis=1).groupby('time').mean()
    rep_list = [i for i in range(0, 300, 25)]
    if OPT == 'c':
        data = [simple_boot(df, sim, stat, chunk_size=i) for i in rep_list]
    if OPT == 'r':
        data = [simple_boot(df, sim, stat, reps=i) for i in rep_list]
    print(stat, terr, sim, 'complete')
    boot_vioplot(data, terr, stat, sim, rep_list)

    # OPT = 'chunk' # 'reps' or 'chunk'

OPT = 'c' 
TITLES = {'r': 'Percent of single days removed', 
          'c': 'Percent of data chunk removed'}
EXP = Experiment('/home/hma000/accomatic-web/tests/test_data/toml/MAR_NWT.toml')
# for i in set(list(EXP.terr_dict().values())):
#     bs_threshold_terr_exp(stat='RMSE', terr=i,  sim='ens')

