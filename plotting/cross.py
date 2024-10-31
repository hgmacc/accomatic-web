import sys

sys.path.append("/home/hma000/accomatic-web/")

import pickle

import matplotlib.pyplot as plt
import pandas as pd
from accomatic.Experiment import *
from accomatic.Stats import r_score
from plotting.box import get_model

idx = pd.IndexSlice


def cross_plot(exp_gst, exp_50, stat, save=True):
    fig_heat = plt.subplots(figsize=(6, 6))
    for mod in exp_gst.columns:
        exp_gst[mod] = [np.mean(cell.arr) for cell in list(exp_gst[mod].values)]
        exp_50[mod] = [np.mean(cell.arr) for cell in list(exp_50[mod].values)]
        plt.scatter(
            exp_gst[mod],
            exp_gst[mod],
            s=5,
            label=f"{get_model[mod]} $r$: {r_score(exp_gst[mod], exp_gst[mod])}",
        )
    plt.legend(loc="best", fontsize="x-small")
    plt.xlabel(f"{stat} at 0.10 m")
    plt.ylabel(f"{stat} at 0.50 m")
    plt.tight_layout()
    if save:
        plt.savefig(f"/home/hma000/accomatic-web/plotting/out/cross/cross_{stat}.png")


if __name__ == "__main__":
    gst = "/home/hma000/accomatic-web/plotting/data/pickles/09May_0.1_0.pickle"
    with open(gst, "rb") as f_gst:
        exp_gst = pickle.load(f_gst)

    cm50 = "/home/hma000/accomatic-web/plotting/data/pickles/29Mar_0.5_0.pickle"
    with open(cm50, "rb") as f_50:
        exp_50 = pickle.load(f_50)

    for stat in exp_gst.stat_list:
        cross_plot(
            exp_gst.results.loc[idx[["res"], 3, :, stat]].droplevel(["mode", "stat"]),
            exp_50.results.loc[idx[["res"], 3, :, stat]].droplevel(["mode", "stat"]),
            stat=stat,
            save=True,
        )
