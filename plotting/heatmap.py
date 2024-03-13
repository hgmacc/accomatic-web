import pickle

import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "25"

import pandas as pd
import sys


sys.path.append("/home/hma000/accomatic-web/accomatic/")
from Stats import rank_distribution, bias_distribution
from Experiment import *
from plotting.box import get_model


def heat(exp, terr="", stat=""):

    df = rank_distribution(exp, terr=terr, stat=stat)
    df = df.round(2)
    df.index = [get_model[mod] for mod in df.index]
    c, size = "YlOrRd", (8, 8)
    if stat == "BIAS":
        c, size = "bwr", (3, 8)
    fig_heat, ax = plt.subplots(figsize=size)
    sns.set(font_scale=2)

    sns.heatmap(
        df,
        annot=True,
        cbar=False,
        vmin=0,
        vmax=1,
        cmap=c,
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0, labelsize=24)

    plt.tight_layout()
    plt.savefig(f"/home/hma000/accomatic-web/plotting/out/heat/heatmap{terr}{stat}.png")
    plt.clf()
