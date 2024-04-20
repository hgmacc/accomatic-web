import pickle
import sys


sys.path.append("/home/hma000/accomatic-web/accomatic/")
from Stats import rank_distribution, bias_distribution
from Experiment import *

import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"


get_colour = {
    "obs": "#000000",
    "merra2": "#F3700E",
    "jra55": "#F50B00",
    "era5": "#008080",
    "ens": "#59473c",
}

terr_desc = [
    "PEATLAND",
    "COARSE HILLTOP",
    "FINE HILLTOP",
    "SNOWDRIFT",
    "HOR. ROCK",
]


def timeseries(exp, sites="", save=False):

    terr_dict = dict(zip(exp.sites_list, [terr_desc[t - 1] for t in exp.terr_list]))

    if sites == "":
        sites = exp.sites_list
    for site in sites:
        df = exp.obs(site).join(exp.mod(site))
        df.index.name = "time"
        fig_heat, ax = plt.subplots(figsize=(15, 10))
        for mod in df.columns:
            plt.plot(df[mod], label=mod, c=get_colour[mod])
        locs, labels = plt.xticks()
        plt.xticks(locs[::2], labels[::2])
        plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5, zorder=-1)

        plt.title(f"{terr_dict[site]}: {site}")
        plt.legend(fontsize="x-small", loc="upper right")
        if save:
            plt.tight_layout()
            plt.savefig(
                f"/home/hma000/accomatic-web/plotting/out/ts/{terr_dict[site]}_{site}_gst.png"
            )
        plt.clf()
        plt.close()


if __name__ == "__main__":
    pth = "/home/hma000/accomatic-web/data/pickles/final_wee.pickle"
    with open(pth, "rb") as f_gst:
        exp = pickle.load(f_gst)
        timeseries(exp, save=True)
