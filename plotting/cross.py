import sys

sys.path.append("/home/hma000/accomatic-web/accomatic/")

import pickle

import matplotlib.pyplot as plt
import pandas as pd
from Experiment import *
from Stats import r_score
from plotting.box import get_model

idx = pd.IndexSlice


def cross_plot(exp_gst, exp_50, save=False):
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
    plt.xlabel(f"MAE at 0.10 m")
    plt.ylabel(f"MAE at 0.50 m")
    plt.tight_layout()
    if save:
        plt.savefig(f"/home/hma000/accomatic-web/plotting/out/cross_{stat}.png")


if __name__ == "__main__":
    gst = "/home/hma000/accomatic-web/data/pickles/06Feb_0.1_0.pickle"
    with open(gst, "rb") as f_gst:
        exp_gst = pickle.load(f_gst)

    cm50 = "/home/hma000/accomatic-web/data/pickles/06Feb_0.5_0.pickle"
    with open(cm50, "rb") as f_50:
        exp_50 = pickle.load(f_50)

    for stat in exp_gst.stat_list:
        cross_plot(
            exp_gst.results.loc[idx[["res"], :, :, stat]].droplevel(["mode", "stat"]),
            exp_50.results.loc[idx[["res"], :, :, stat]].droplevel(["mode", "stat"]),
            save=stat,
        )

# plt.plot(xlims, ylims, color="k", zorder=0)
# # XY SCATTER PLOT: 10 & 100
# plt.subplot(212)
# ax2.set_aspect("equal")
# for mod, col in zip(models, palette):
#     s = f"{mod.upper()} $r$={np.corrcoef(df100.loc[mod][10],df100.loc[mod][100])[0][1]:.2f}"
#     plt.scatter(df100.loc[mod][10], df100.loc[mod][100], s=5, c=col, label=s)
# # plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
# plt.xlabel("MAE 0.1 m")
# plt.ylabel("MAE 1.0 m")

# plt.plot(xlims, ylims, color="k", zorder=0)
# plt.legend(fontsize="x-small")

# fig.savefig(f"plane_plot_rank.png")

# fig.clf()
# plt.close(fig)
