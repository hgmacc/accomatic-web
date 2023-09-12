import random

import matplotlib.font_manager
import matplotlib.image as image
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
from static.statistics_helper import rank_shifting_for_heatmap, time_code_months

# from accomatic.NcReader import *
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
import os
from Experiment import *

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"

PLOT_PTH = "/home/hma000/accomatic-web/plots"


def hex_to_RGB(hex_str):
    """#FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i : i + 2], 16) for i in range(1, 6, 2)]


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    c1_rgb = np.array(hex_to_RGB(c1)) / 255
    c2_rgb = np.array(hex_to_RGB(c2)) / 255
    mix_pcts = [x / (n - 1) for x in range(n)]
    rgb_colors = [((1 - mix) * c1_rgb + (mix * c2_rgb)) for mix in mix_pcts]
    return [
        "#" + "".join([format(int(round(val * 255)), "02x") for val in item])
        for item in rgb_colors
    ]


palette_list = ["#527206", "#584538", "#008184", "#F50400", "15e2d0"]


def get_colour(f):
    if "mer" in f:
        return "#F3700E"
    if "era" in f:
        return "#008080"
    if "jra" in f:
        return "#F50B00"
    if "ens" in f:
        return "#59473c"
    else:
        return f"{f} is not a pth with a colour."


def violin_helper_reorder_data(data, stat):
    data["rank"] = ["{0:.3}".format(np.nanmean(i.v)) for i in data[stat]]
    return data.sort_values(by=["rank"])


def boot_vioplot(e, title=""):
    # site, stat, sim, label,

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

    if title == "":
        import time

        title = f"vio_{time.time()}"
        print(title)

    legend_handles = [
        f"({a})" for i, a in zip(data.sim.to_list(), data["rank"].tolist())
    ]
    plt.legend(legend_handles, loc="lower left")
    plt.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/{title}.png")
    plt.clf()


def MAE_cross_plots(df):

    var = "rank"
    models = df.sim.unique().tolist()
    df["sett"] = df.szn + df.terr.astype(str)
    xlims, ylims = (df[var].min(), df[var].max()), (df[var].min(), df[var].max())
    df = df.set_index(["sim", "sett", "depth"])[var].unstack("depth")

    df50, df100 = df[[10, 50]].dropna(), df[[10, 100]].dropna()
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10), squeeze=True)

    palette = ["#59473c", "#008080", "#F3700E", "#F50B00"]

    # XY SCATTER PLOT: 10 and 50
    plt.subplot(211)
    ax1.set_aspect("equal")

    for mod, col in zip(models, palette):
        s = f"{mod.upper()} $r$={np.corrcoef(df50.loc[mod][10],df50.loc[mod][50])[0][1]:.2f}"
        plt.scatter(df50.loc[mod][10], df50.loc[mod][50], s=5, c=col, label=s)

    plt.ylabel("MAE 0.5 m")
    plt.legend(fontsize="x-small")

    plt.plot(xlims, ylims, color="k", zorder=0)

    # XY SCATTER PLOT: 10 & 100
    plt.subplot(212)
    ax2.set_aspect("equal")

    for mod, col in zip(models, palette):
        s = f"{mod.upper()} $r$={np.corrcoef(df100.loc[mod][10],df100.loc[mod][100])[0][1]:.2f}"
        plt.scatter(df100.loc[mod][10], df100.loc[mod][100], s=5, c=col, label=s)

    # plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlabel("MAE 0.1 m")
    plt.ylabel("MAE 1.0 m")

    plt.plot(xlims, ylims, color="k", zorder=0)
    plt.legend(fontsize="x-small")

    fig.savefig(f"plane_plot_rank.png")

    fig.clf()
    plt.close(fig)


def extrapolation_heatmap(df2):
    #         Terrain_Season
    # 0.1         0/3
    # 0.5         1/3
    # 1.0         2/3
    fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(25, 15))
    for simulation, ax in zip(df2.sim.unique(), axs.flat):
        df = df2
        df["szn_no"] = [time_code_months[i][0] for i in df.szn]
        df["szn_terr"] = df["szn_no"].astype(str) + df["terr"].astype(str)
        df["szn_terr"] = df["szn_terr"].astype(int)

        df = df[["sim", "rank", "stat", "depth", "szn_terr"]]
        df = df[df.sim == simulation].drop("sim", axis=1)

        df = df.set_index(["depth", "stat", "szn_terr"]).unstack("szn_terr")
        df = df.groupby("depth", as_index=False).agg(rank_shifting_for_heatmap)
        farmers = df.columns.levels[1]

        z = df.to_numpy()
        ax.imshow(z)

        vegetables = ["0.1", "0.5"]
        ax.set_xticks(np.arange(len(farmers)), labels=farmers, rotation=70)
        ax.set_yticks(np.arange(len(vegetables)), labels=vegetables)
        ax.xaxis.tick_top()  # x axis on top
        ax.xaxis.set_label_position("top")
        ax.set_ylabel(simulation)
        for i in range(len(vegetables)):
            for j in range(len(farmers)):
                text = ax.text(
                    j, i, f"{z[i, j]}/3", ha="center", va="center", color="w"
                )

    plt.savefig("heatmap.png")


def terrain_timeseries(exp):
    # THIS PLOT IS TO SHOW TERRAIN TYPES
    # o = exp.obs()
    # o['terr'] = [exp.terr_dict()[x] for x in o.index.get_level_values(1)]
    # o = o.reset_index()
    o = pd.read_csv("tmp.csv", parse_dates=["time"])

    o = o[o.time.dt.year == 2020]
    fig = plt.figure(figsize=(15, 10))

    sns.lineplot(x="time", y="obs", hue="terr", data=o, palette=palette_list)

    plt.ylabel("Observed Temperature ˚C")
    plt.xlabel("Time")
    plt.legend(title="Terrain Description", labels=exp.terr_desc.values())

    plt.savefig(f"{PLOT_PTH}tmp.png")
