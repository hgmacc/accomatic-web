import getopt
import glob
import os
import re
import sys

from Experiment import *
from NcReader import *
from Stats import *
from Plotting import *


def seasonal_heatmap():
    e = Experiment(arg_input)
    build(e) 
    xy_site_plot(e,'NGO-DD-1005')
    a = pd.read_csv('/home/hma000/accomatic-web/NGO-DD-1005_temp.csv')    
    ens = pd.DataFrame({'sim':['ens' for i in range(5)],
                        'szn':list(a.groupby(['szn'], sort=False).mean().index),
                        'MAE':list(a.groupby(['szn'], sort=False).mean().MAE)})

    a = pd.concat([a, ens]).reset_index(drop=True)
    a['rank'] = a.groupby(['szn']).MAE.rank(method="min").astype(int)
    
    pal = ["#008080", "#F50B00", "#F3700E", "#59473c"]

    x_terrains = ["Annual", "Winter", "Spring", "Summer", "Fall"]
    y_mod = ["ERA5", "JRA55", "MERRA2", "ENSEMBLE"]
    data = np.array([list(a[a.sim==m].MAE) for m in ["era5", "jra55", "merra2", "ens"]])
    data_ranks = np.array([list(a[a.sim==m]['rank']) for m in ["era5", "jra55", "merra2", "ens"]])

    fig, ax = plt.subplots()

    im, cbar = heatmap(data_ranks, y_mod, x_terrains, ax=ax,
                    cmap="YlGn", cbarlabel="Model Ranking")
    #texts = annotate_heatmap(im, data=data, valfmt="{x:.2f}")
    plt.title("MAE Performance of Each Model")
    fig.tight_layout()
    PLOT_PTH = '/home/hma000/accomatic-web/tests/plots/29MAR_heatmap/'
    plt.savefig(f'{PLOT_PTH}heatmap_plot.png', dpi=300) 
    

def terr_heatmap():
    a = pd.read_csv("/home/hma000/accomatic-web/30mar_terr_heatmap.csv")
    ens = a.groupby(['site'], sort=False).mean().reset_index(drop=False)
    ens['sim'] = 'ens'
    a = pd.concat([a, ens]).sort_values('site')
    a = a.groupby(['sim','terr']).mean().reset_index(drop=False)
    a['rank'] = a.groupby(['terr']).MAE.rank(method="min").astype(int)
    
    # annotate = a.MAE
    # ranking = a.terr
    
    pal = ["#008080", "#F50B00", "#F3700E", "#59473c"]
    x_terrains = [str(i) for i in a.terr.unique()]
    y_mod = ["ERA5", "JRA55", "MERRA2", "ENSEMBLE"]
    data = np.array([list(a[a.sim==m].MAE) for m in ["era5", "jra55", "merra2", "ens"]])
    data_ranks = np.array([list(a[a.sim==m]['rank']) for m in ["era5", "jra55", "merra2", "ens"]])

    fig, ax = plt.subplots()

    im, cbar = heatmap(data_ranks, y_mod, x_terrains, ax=ax,
                    cmap="YlGn", cbarlabel="Model Ranking")
    texts = annotate_heatmap(im, data=data, valfmt="{x:.2f}")
    plt.title("MAE Performance of Each Model")
    fig.tight_layout()
    PLOT_PTH = '/home/hma000/accomatic-web/tests/plots/29MAR_heatmap/'
    plt.savefig(f'{PLOT_PTH}terr_heatmap_plot.png', dpi=300) 
    
terr_heatmap()