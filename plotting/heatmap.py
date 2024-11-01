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


def heat(exp, terr="", stat="", szn="", save=False):

    df = rank_distribution(exp, terr=terr, stat=stat, szn=szn)
    df = df.round(2)
    df.index = [get_model[mod] for mod in df.index]
    c, size = sns.blend_palette(["#FFFFFF", "#000000"], as_cmap=True), (
        8,
        8,
    )
    if stat == "BIAS":
        c, size = "bwr", (3, 8)
    fig_heat, ax = plt.subplots(figsize=size)
    sns.set_theme(font_scale=2)

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
    if save:
        plt.savefig(
            f"/home/hma000/accomatic-web/plotting/out/heat/{exp.depth}/heatmap{terr}{stat}{szn}.png"
        )
    else:
        return ax
    plt.clf()


try:
    arg = sys.argv[1]
except IndexError:
    arg = False

if __name__ == "__main__":
    pth = "/home/hma000/accomatic-web/plotting/data/pickles/24May_0.5_0.pickle"
    # pth = "/home/hma000/accomatic-web/plotting/data/pickles/09May_0.1_0.pickle"
    with open(pth, "rb") as f_gst:
        exp = pickle.load(f_gst)

    heat(exp, terr=1, stat="MAE", szn="MAY", save=True)
    heat(exp, terr=1, stat="MAE", szn="JAN", save=True)
    heat(exp, terr=1, stat="MAE", save=True)
    heat(exp, terr=1, stat="R", save=True)
    heat(exp, terr=1, stat="BIAS", save=True)

    for s in exp.stat_list:
        for t in set(exp.terr_list):
            heat(exp, terr=t, stat=s, save=True)
            if s == "MAE":
                heat(exp, terr=t, save=True)
        heat(exp, stat=s, save=True)
    heat(exp, save=True)
