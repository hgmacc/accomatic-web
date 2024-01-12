import pickle

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "14"

import pandas as pd
import sys

sys.path.append("../")
from accomatic.Stats import rank_distribution
from accomatic.Experiment import *


def bias_heatmap(exp, save=True, title="bias"):
    df = exp.bias_dist
    fig_bias, ax = plt.subplots(figsize=(4.5, 6))

    sns.heatmap(
        exp.bias_dist.values,
        vmin=0,
        vmax=1,
        cmap="RdBu_r",
        annot=True,
        fmt=".2g",
        linewidths=1,
        linecolor="white",
        cbar=True,
        cbar_kws={
            "label": "Proportion of positive bias",
            "ticks": [0.0, 0.5, 1.0],
        },
        square=True,
        xticklabels=["WARM BIAS"],
        yticklabels=[i.upper() for i in list(exp.bias_dist.index)],
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    if save:
        plt.savefig(f"/home/hma000/accomatic-web/plotting/out/{title}.png")
    return fig_bias


def rank_dist_heatmap(rank_dist, save=True, title="heatmap"):
    fig_heat, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(
        rank_dist.values,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        annot=True,
        fmt=".2g",
        linewidths=1,
        linecolor="white",
        cbar=True,
        cbar_kws={  # Legend bar
            "label": "Proportion of instances occupying rank",
            "shrink": 0.75,
            "location": "bottom",
            "pad": 0.1,
            "ticks": [0.0, 0.5, 1.0],
        },
        square=True,
        xticklabels=[i.upper() for i in list(rank_dist.columns)],
        yticklabels=["Model 1", "Model 2", "Model 3", "Ensemble"],
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    if save:
        plt.savefig(f"/home/hma000/accomatic-web/plotting/out/{title}.png")
    return fig_heat


import sys
import pickle

try:
    test = sys.argv[1]
    if test == "-t":
        pth = "/home/hma000/accomatic-web/data/pickles/2024-01-12_results.pickle"
        print(pth)
        with open(pth, "rb") as f_gst:
            exp = pickle.load(f_gst)

        exp.rank_dist = rank_distribution(exp)

        bias_heatmap(exp, save=True)
        rank_dist_heatmap(exp.rank_dist, save=True)

except IndexError:
    pass
