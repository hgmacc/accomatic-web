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
    if site == 'all sites':
        df = e.mod().join(e.obs()).dropna()
        df = df.groupby('time').mean()
    else:
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
    print('One done')
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

def boot_vioplot(data, site, stat, sim, label):
    fig, ax = plt.subplots(figsize=(len(data)+2, 10))
    bp = ax.violinplot(data, showmeans=True)

    for patch, color in zip(bp['bodies'], get_color_gradient("#b3e0dc", "#036c5f", len(label))):
        patch.set_facecolor(color) 
        patch.set_alpha(1.0)
        
    for partname in ('cbars','cmins','cmaxes','cmeans'):
        vp = bp[partname]
        vp.set_edgecolor('#000000')
        vp.set_linewidth(1)
     
    # This is the most annoying line of code I've ever written. 
    ax.set_xticks([i for i in range(0, len(label), 2)], labels=[str(i) for i in label[::2]])
    
    ax.set_title(f"{sim} at {site}")
    ax.set_xlabel(TITLES[OPT])
    ax.set_ylabel(stat)
    if stat != 'R': ax.set_ylim(bottom=0, top=1.5)
    plt.savefig(f'/home/hma000/accomatic-web/accomatic/prototype_bootstrap/plots/vio_{OPT}_{stat}_{site}_rep.png')
    plt.clf()

def bs_threshold_exp(stat, site='all sites', sim='ens'): 
    df = get_data(site)[[sim, 'obs']]    
    rep_list = [i for i in range(0, 1000, 50)]
    if OPT == 'c':
        data = [simple_boot(df, sim, stat, chunk_size=i) for i in rep_list]
    if OPT == 'r':
        data = [simple_boot(df, sim, stat, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, rep_list)
    
def boot_terrain(data, site, stat, sim, label, terr):    
    sites = exp.sites_list
    terr = exp.terr_list

    terr_dict = {sites[i]: terr[i] for i in range(len(sites))}
    num_plots = len(set(terr))

    fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=(10, num_plots+10))#, squeeze=True)
    fig.suptitle('Missing data plot for each terrain type.')

    df = read_nc('/home/hma000/accomatic-web/tests/test_data/nc/ykl_obs.nc', avg=True)
    
    # Pull out only dat
    terr_list = []
    for i in df.index.get_level_values(1):
        try: terr_list.append(terr_dict[i])
        except KeyError:
            terr_list.append(-1)
    
    df['terrain'] = terr_list

    a = df[df.terrain == str(i)].drop(["terrain"], axis=1)
    a = a.rename_axis(index=('time', None))
    a = a.obs.unstack(level=1)

def bs_threshold_terr_exp(stat, terr, site='all sites', sim='ens'): 
    df = get_data(site)[[sim, 'obs']]    
    rep_list = [i for i in range(0, 1000, 50)]
    if OPT == 'c':
        data = [simple_boot(df, sim, stat, chunk_size=i) for i in rep_list]
    if OPT == 'r':
        data = [simple_boot(df, sim, stat, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, rep_list)
  
# OPT = 'chunk' # 'reps' or 'chunk'
OPT = 'c' 
TITLES = {'r': 'one single day removed n times', 
          'c': 'n consective days removed once.'}


#bs_threshold_exp(stat='R', site='all sites', sim='jra55')
#bs_threshold_terr_exp(stat='R', terr='1', site='all sites', sim='jra55')
exp = Experiment('/home/hma000/accomatic-web/tests/test_data/toml/MAR_NWT.toml')

for i in exp.terr_dict():
    print(i)
sys.exit()
exp = Experiment('/home/hma000/accomatic-web/tests/test_data/toml/MAR_NWT.toml')
for i in exp.terr_dict:
    bs_threshold_terr_exp(stat='R', terr=int(i), site='all sites', sim='jra55')
