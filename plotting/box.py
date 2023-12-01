import os
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

import pandas as pd

sys.path.append("../")
from accomatic.Experiment import *
import accomatic

get_colour = {
    "obs": "#000000",
    "merra2": "#F3700E",
    "jra55": "#F50B00",
    "era5": "#008080",
    "ens": "#59473c",
}


def violin_helper_reorder_data(data, stat):
    data["rank"] = ["{0:.3}".format(np.nanmean(i.v)) for i in data[stat]]
    return data.sort_values(by=["rank"])


def boot_vioplot(e, save=True):
    # site, stat, sim, label

    stat = "MAE"
    if type(e) == Experiment():
        data = e.res(sett=["sim"])
        data = violin_helper_reorder_data(data, stat)

        label = data.sim.to_list()
        data_arr = np.array([i.v for i in data[stat].to_list()])
    else:
        data_arr = e

    fig, ax = plt.subplots(figsize=(len(data_arr) + 4, 8))

    bp = ax.violinplot(data_arr.T, showmeans=True)

    for patch, mod in zip(bp["bodies"], label):
        patch.set_facecolor(get_colour(mod))
        patch.set_alpha(1.0)

    for partname in ("cbars", "cmins", "cmaxes", "cmeans"):
        vp = bp[partname]
        vp.set_edgecolor("#000000")
        vp.set_linewidth(1)

    ax.set_ylabel(stat)
    ax.set_ylim(bottom=0, top=4)

    legend_handles = [
        f"({a})" for i, a in zip(data.sim.to_list(), data["rank"].tolist())
    ]
    plt.legend(legend_handles, loc="lower left")
    if save:
        plt.savefig(f"plotting/out/violin.png")
    return fig


def boxplot(exp, stat="", terr="", save=True, bw=False):
    """
    Function takes and df with x columns
    If you want colour coordination with models,
    column name must start with "mod_"

    i.e. "era5_jan_rock" -> c="#1ce1ce"

    Box and whisker plots results
    Will always plot in order: Best -> Worst
    """
    if stat == "":
        stat = list(exp.acco_list)
    if terr == "":
        terr = list(set(exp.terr_list))

    # Selecting data from results
    idx = pd.IndexSlice
    df = exp.results.loc[idx[["res"], terr, :, stat]].droplevel("mode")

    # Pulling out arrays into 1D arr for each mod
    data = {}
    for mod in df.columns:
        data[mod] = np.concatenate([cell.arr for cell in df[mod]], axis=0)

    # Building figure
    fig_box, ax = plt.subplots(figsize=(1 * len(data.keys()), 8))
    bp = ax.boxplot(data.values(), whis=1.5, sym="", patch_artist=True, showmeans=True)
    ax.set_xticklabels(["Model 1", "Model 2", "Model 3", "Ensemble"])
    ax.set_ylim(-25, 25)

    if stat == "WILL":
        plt.gca().yaxis.set_major_formatter(
            StrMethodFormatter("{x:,.2f}")
        )  # 2 decimal places

    # Setting the color of each box
    for patch, col in zip(bp["boxes"], list(data.keys())):
        if not bw:  # If colour:
            patch.set_facecolor(get_colour[col])
        patch.set_facecolor("#FFFFFF")

    # making the mean a nice little black diamond
    for mean in bp["means"]:
        mean.set_markerfacecolor("#000000")
        mean.set_markeredgecolor("#000000")
        mean.set_marker("D")

    # Setting median line to black
    for median in bp["medians"]:
        median.set_color("#000000")

    if not bw:  # If colour:
        legend_elements = []
        for c in data.keys():
            legend_elements.append(
                Patch(facecolor=get_colour[c], edgecolor="#000000", label=c.upper())
            )

    if save:
        plt.savefig(
            f"/home/hma000/accomatic-web/plotting/out/box/box_{terr}_{stat}.png"
        )
    return fig_box


import pickle

pth = "/home/hma000/accomatic-web/data/pickles/NOV30_bs1000_d01.pickle"
with open(pth, "rb") as f:
    exp = pickle.load(f)

from plotting.box import *

for i in range(6):
    fig = boxplot(exp, stat="BIAS", bw=False, terr=i + 1, save=True)
