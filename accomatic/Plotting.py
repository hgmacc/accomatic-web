import random

import matplotlib.font_manager
import matplotlib.image as image
from matplotlib.patches import Patch

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


palette_list = ["#59473c", "#F50B00", "#008080", "#F3700E", "#15e2d0"]


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
        return False


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


def cluster_timeseries(exp):
    ######## ALL OBS PLOT #######################
    o_clusters = exp.obs().obs.unstack(level=1)
    o_clusters.index = pd.to_datetime(o_clusters.index)

    o_clusters["day-month"] = o_clusters.index.strftime("%m-%d")
    o_clusters = o_clusters.groupby(["day-month"]).mean()

    yk_col = [col for col in o_clusters.columns if "YK" in col]
    kdi_col = [col for col in o_clusters.columns if "KDI" in col]
    ldg_col = [col for col in o_clusters.columns if "NGO" in col]
    ldg_col.extend([col for col in o_clusters.columns if "ROCK" in col])

    fig = plt.figure(figsize=(6, 6))

    l = []
    for clust_cols, clust_name in zip([kdi_col, ldg_col, yk_col], ["KDI", "LDG", "YK"]):
        tmp = (
            o_clusters[clust_cols]
            .stack()
            .reset_index(drop=False)
            .rename(columns={0: "obs"})
        )
        tmp["cluster"] = clust_name
        l.append(tmp)

    df = pd.concat(l)

    sns.lineplot(
        data=df,
        x="day-month",
        y="obs",
        hue="cluster",
        palette=["#F50B00", "#F3700E", "#008080"],
        legend=False,
    )

    plt.legend(
        handles=[
            Patch(facecolor=hex, edgecolor=hex, label=cluster)
            for hex, cluster in zip(
                ["#F50B00", "#F3700E", "#008080"],
                ["KDI", "Lac de Gras (LDG)", "Yellowknife (YK)"],
            )
        ],
        loc="upper right",
        fontsize="small",
    )
    plt.xticks([], [])
    plt.xlabel("")
    plt.ylabel("Observed Temperature ˚C")

    plt.savefig("/home/hma000/accomatic-web/plots/workflow/all_obs.png")


def terrain_timeseries(exp):
    ######## TERRAIN PLOT #######################

    o = exp.obs().reset_index(drop=False)
    o.level_0 = pd.to_datetime(o.level_0)

    fig = plt.figure(figsize=(8, 8))
    o["day-month"] = o.level_0.dt.strftime("%m-%d")
    o = o.groupby(["day-month", "sitename"]).mean().drop(columns="level_0")
    o["terr"] = [exp.terr_dict()[x] for x in o.index.get_level_values(1)]

    sns.lineplot(
        data=o.dropna(),
        x="day-month",
        y="obs",
        hue="terr",
        palette=palette_list,
    )
    
    months = ["JAN", "MAR", "MAY", "JUL", "SEP", "NOV"]
    plt.xticks(ticks=range(1, 365, 62), labels=months)
    plt.ylabel("Observed Temperature ˚C")
    plt.xlabel("Time")

    # Have to do this bc sns.lineplot legend is weird as hell
    legend_elements = [
        Patch(facecolor=c, edgecolor=c, label=des)
        for c, des in zip(palette_list, exp.terr_desc.values())
    ]
    plt.legend(handles=legend_elements, loc="upper right", fontsize="small")


def one_terr(exp):

    fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(10, 8))

    ######## ALL OBS PLOT #######################
    plt.subplot(421)

    o_clusters = exp.obs().obs.unstack(level=1)

    yk_col = [col for col in o_clusters.columns if "YK" in col]
    kdi_col = [col for col in o_clusters.columns if "KDI" in col]
    ldg_col = [col for col in o_clusters.columns if "NGO" in col]
    ldg_col.extend([col for col in o_clusters.columns if "ROCK" in col])

    for cluster, hex in zip(
        [kdi_col, ldg_col, yk_col], ["#F50B00", "#F3700E", "#008080"]
    ):
        cluster_df = (
            o_clusters[cluster]
            .stack()
            .reset_index(drop=False)
            .rename(columns={0: "obs"})
        )

        sns.lineplot(
            data=cluster_df,
            x="level_0",
            y="obs",
            palette=sns.light_palette(
                hex,
                input="rgb",
                n_colors=len(cluster_df.sitename.unique()),
            ),
            hue="sitename",
            legend=False,
            linewidth=0.5,
        )

    plt.subplot(4, 2, 3)

    ######## TERRAIN PLOT #######################
    o = exp.obs().reset_index(drop=False)
    o.level_0 = pd.to_datetime(o.level_0)
    o["day-month"] = o.level_0.dt.strftime("%m-%d")
    o = o.groupby(["day-month", "sitename"]).mean().drop(columns="level_0")
    o["terr"] = [exp.terr_dict()[x] for x in o.index.get_level_values(1)]

    sns.lineplot(
        data=o.dropna(),
        x="day-month",
        y="obs",
        hue="terr",
        palette=palette_list,
    )
    months = ["JAN", "MAR", "MAY", "JUL", "SEP", "NOV"]
    plt.xticks(ticks=range(1, 365, 62), labels=months)
    plt.ylabel("Observed Temperature ˚C")
    plt.xlabel("Time")

    # Have to do this bc sns.lineplot legend is weird as hell
    legend_elements = [
        Patch(facecolor=c, edgecolor=c, label=des)
        for c, des in zip(palette_list, exp.terr_desc.values())
    ]
    plt.legend(handles=legend_elements)

    for terrain in exp.terr_list:
        plt.subplot(4, 2, terrain + 3)
        # Plot one terrain; all 12 months
        data = exp.data[terrain]
        # szn = list_of_df
        for season in data.keys():
            df = pd.concat(data[season])
            df = df.reset_index()
            df.Date = pd.to_datetime(df.Date).apply(lambda x: x.strftime("%m %d"))
            if terrain == 1:
                i = True
            else:
                i = False
            sns.lineplot(data=df, x="Date", y="obs", label=season, legend=i)
            plt.title(exp.terr_desc[terrain])
    plt.savefig("/home/hma000/accomatic-web/plots/workflow/tmp.png")
