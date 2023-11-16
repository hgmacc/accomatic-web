import pickle
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "24"

import pandas as pd


def bias_heatmap(exp):
    df = exp.bias_dist
    fig_bias, ax = plt.subplots(figsize=(9, 12))

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

    plt.savefig("/home/hma000/accomatic-web/plotting/out/bias.png")


def rank_dist_heatmap(exp):
    fig_heat, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(
        exp.rank_dist.values,
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
        xticklabels=[i.upper() for i in list(exp.rank_dist.columns)],
        yticklabels=[i.upper() for i in list(exp.rank_dist.index)],
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    plt.savefig("/home/hma000/accomatic-web/plotting/out/heatmap.png")


"""
To run: 

exp = Experiment("/home/hma000/accomatic-web/data/toml/run.toml")
build(exp)
df = exp.results()
spiderplot(df) 

plot in: out/heatmap.png

"""
