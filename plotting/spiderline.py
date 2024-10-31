import sys
import matplotlib as plt
import pandas as pd

sys.path.append("/home/hma000/accomatic-web/accomatic/")
from Experiment import *
from plotting.box import *
from Stats import *
from matplotlib.ticker import FormatStrFormatter
import numpy as np

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "12"

terr_desc = [
    "Peatland",
    "Coarse Hilltop",
    "Fine Hilltop",
    "Snowdrift",
    # "Hor. Rock",
    "MEAN",
]


def spider(exp, save=False):
    fig, ax = plt.subplots(
        figsize=(14, 10), sharex=True, ncols=3, nrows=len(list(set(exp.terr_list))) + 1
    )

    idx = pd.IndexSlice

    terrains = list(set(exp.terr_list))
    stats = exp.stat_list
    limits = [(0, 15), (-10, 7.5), (0, 1)]
    for row in range(len(terrains) + 1):
        print(row)
        for col in range(3):
            if row == 4:  # MEAN bottom row
                print("row == 4")
                df = exp.results.loc[idx[["res"], :, :, stats[col]]].droplevel("mode")
            else:
                df = exp.results.loc[
                    idx[["res"], terrains[row], :, stats[col]]
                ].droplevel("mode")

            for reanalysis in df.columns:
                if row == 4:  # MEAN bottom row
                    print("row == 4")

                    df[reanalysis] = [
                        np.mean(cell.arr) for cell in list(df[reanalysis].values)
                    ]
                    data = df[reanalysis].groupby(["szn"]).mean().reindex(exp.szn_list)
                    ax[0, col].set_title(stats[col])
                else:
                    data = [np.mean(cell.arr) for cell in list(df[reanalysis].values)]

                ax[row, col].plot(
                    exp.szn_list,
                    data,
                    label=get_model[reanalysis],
                    c=get_colour[reanalysis],
                )
                ax[row, col].set_ylim(limits[col])
            ax[row, 0].set_ylabel(terr_desc[row])
    ax[0, 0].legend()
    locs, labels = plt.xticks()
    plt.xticks(locs[::2], labels[::2])
    plt.tight_layout()
    if save:
        plt.savefig("/home/hma000/accomatic-web/plotting/out/spider05.png")
    else:
        return ax


if __name__ == "__main__":
    pth = "/home/hma000/accomatic-web/data/pickles/24May_0.5_0.pickle"
    with open(pth, "rb") as f_gst:
        exp = pickle.load(f_gst)
    spider(exp, save=True)
