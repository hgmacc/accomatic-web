import os
import sys
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"
sys.path.append("/home/hma000/accomatic-web/accomatic/")
from Experiment import *

get_colour = {
    "obs": "#000000",
    "era5": "#008080",
    "jra55": "#F50B00",
    "merra2": "#F3700E",
    "ens": "#59473c",
    r"$M_1$": "#8fbdbc",  # ERA5
    r"$M_2$": "#f28e89",  # JRA55
    r"$M_3$": "#f4c18e",  # MERRA2
    r"$M_E$": "#a39e97",  # Ensemble
}

get_model = {
    "era5": r"$M_1$",
    "jra55": r"$M_2$",
    "merra2": r"$M_3$",
    "ens": r"$M_E$",
}

stat_bounds = {"BIAS": [-13, 10], "R": [-1, 1], "MAE": [0, 12.5]}


def boxplot(data, bounds=False, save=False, bw=False):
    """
    Data = []
    """
    # Building figure
    fig_box, ax = plt.subplots(figsize=(1.1 * len(data.keys()), 8))
    bp = ax.boxplot(data.values(), whis=1.5, sym="", patch_artist=True, showmeans=True)
    ax.set_xticklabels(data.keys())

    if bounds:
        ax.set_ylim(bounds[0], bounds[1])

    plt.gca().yaxis.set_major_formatter(StrMethodFormatter("{x:,.2f}"))

    # Setting the color of each box
    for patch, mod in zip(bp["boxes"], data.keys()):
        if bw:
            patch.set_facecolor("#FFFFFF")
        else:
            patch.set_facecolor(get_colour[mod.split("-")[0]])

    # making the mean a nice little black diamond
    for mean in bp["means"]:
        mean.set_markerfacecolor("#000000")
        mean.set_markeredgecolor("#000000")
        mean.set_marker("D")

    # Setting median line to black
    for median in bp["medians"]:
        median.set_color("#000000")

    plt.tight_layout()
    if save:
        plt.savefig(save)
        print(f"New plot saved to: {save}")
    else:
        return ax


def one_exp(exp, szn="", stat="", terr="", save=False):
    """
    Function organizes data for four boxplots.
    Can represent x1 stat, x1 terrain, x 1 season at a time OR aggregated values.
    Max 1 boxplot per mode though.
    """
    pth, bounds = False, False
    if save:
        pth = f"/home/hma000/accomatic-web/plotting/out/box/box{stat}{terr}{szn}.png"
    if szn == "":
        szn = list(set(exp.szn_list))
    if stat == "":
        print("No! We need a stat for a boxplot.")
        sys.exit()
    if terr == "":
        terr = list(set(exp.terr_list))

    bounds = stat_bounds[stat]

    # Selecting data from results
    idx = pd.IndexSlice
    df = exp.results.loc[idx[["res"], terr, szn, stat]].droplevel("mode")

    # Pulling out arrays into 1D arr for each mod
    data = {}
    for mod in df.columns:
        name = get_model[mod]
        data[name] = np.concatenate([cell.arr for cell in df[mod]], axis=0)
    # Building figure
    boxplot(data, bounds=bounds, save=pth)


try:
    arg = sys.argv[1]
except IndexError:
    arg = False

if __name__ == "__main__":
    pth = "data/pickles/final_wee.pickle"
    with open(pth, "rb") as f_gst:
        exp = pickle.load(f_gst)

    for s in exp.stat_list:
        for t in set(exp.terr_list):
            one_exp(exp, terr=t, stat=s, save=True)
        one_exp(exp, stat=s, save=True)
