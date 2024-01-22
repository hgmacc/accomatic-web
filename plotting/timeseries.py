import pickle
import sys

sys.path.append("../")


import matplotlib.pyplot as plt
from accomatic.Experiment import *

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


def timeseries(exp, sites=""):
    if sites == "":
        sites = exp.sites_list
    for site in sites:
        df = exp.obs(site).join(exp.mod(site))
        df.index.name = "time"
        fig_heat, ax = plt.subplots(figsize=(10, 10))
        for mod in df.columns:
            plt.plot(df[mod], label=mod, c=get_colour[mod])
        locs, labels = plt.xticks()
        plt.xticks(locs[::2], labels[::2])
        plt.title(site)
        plt.legend(fontsize="x-small", loc="upper right")
        plt.savefig(f"/home/hma000/accomatic-web/plotting/out/ts/{site}.png")
        plt.clf()
        plt.close()


if sys.argv[0] == "t":
    pth = "/home/hma000/accomatic-web/plotting/plotting.pickle"
    with open(pth, "rb") as f_gst:
        exp = pickle.load(f_gst)

    timeseries(exp, save=True)
