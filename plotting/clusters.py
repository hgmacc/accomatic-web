import pickle
import sys
import seaborn as sns
from matplotlib.patches import Patch

sys.path.append("../")


import matplotlib.pyplot as plt
from accomatic.Experiment import *

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"


def terrain_timeseries(exp, save=False):
    ######## TERRAIN PLOT #######################
    fig, axs = plt.subplots(5, 1, sharey=True, sharex=True, figsize=(36, 12))

    o = exp.obs().reset_index(drop=False)

    obs_sites = list(o.sitename.unique())
    exp_sites = exp.sites_list

    o.level_0 = pd.to_datetime(o.level_0)

    o["day-month"] = o.level_0.dt.strftime("%m-%d")
    o["year"] = o.level_0.dt.strftime("%y")
    o.set_index("day-month", inplace=True)
    o = o.drop(columns="level_0")

    # o = o.groupby(["day-month", "sitename"]).mean().drop(columns="level_0")
    o["terr"] = [exp.terr_dict()[x] for x in o.sitename]

    terr_desc = [
        "PEATLAND",
        "COURSE_HILLTOP",
        "FINE_HILLTOP",
        "SNOWDRIFT",
        "HOR_ROCK",
    ]

    terr_dict = dict(zip(range(1, 7), terr_desc))
    o.terr = [terr_dict[i] for i in o.terr]
    for i in terr_dict.keys():
        plt.subplot(5, 1, i)

        sns.lineplot(
            data=o[o.terr == terr_dict[i]].dropna(),
            x="day-month",
            y="obs",
            hue="sitename",
            legend=True,
        )
        plt.title(terr_dict[i])
        plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5, zorder=-1)
        l = plt.legend(
            title="",
            loc="upper left",
            fontsize="xx-small",
        )
        if i == 5:
            months = ["JAN", "MAR", "MAY", "JUL", "SEP", "NOV"]
            plt.xticks(ticks=range(1, 365, 62), labels=months)
        plt.xlabel("")
        plt.ylabel("")
    fig.supylabel("Observed Temperature ˚C")
    fig.supxlabel("Time")
    plt.tight_layout()
    if save:
        plt.savefig("/home/hma000/accomatic-web/plotting/out/terrains.png")


def cluster_timeseries(exp, bw=False, save=False):
    ######## ALL OBS PLOT #######################
    o_clusters = exp.obs().obs.unstack(level=1)
    o_clusters.index = pd.to_datetime(o_clusters.index)

    o_clusters["day-month"] = o_clusters.index.strftime("%m-%d")
    o_clusters = o_clusters.groupby(["day-month"]).mean()

    yk_col = [col for col in o_clusters.columns if "YK" in col]
    kdi_col = [col for col in o_clusters.columns if "KDI" in col]
    ldg_col = [col for col in o_clusters.columns if "NGO" in col]
    ldg_col.extend([col for col in o_clusters.columns if "ROCK" in col])

    fig = plt.figure(figsize=(12, 6))

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
        data=df.dropna(),
        x="day-month",
        y="obs",
        hue="cluster",
        palette=["#F50B00", "#F3700E", "#1ce1ce"],
        legend=False,
    )
    # plotting 0˚C line
    df["zero"] = 0
    plt.plot(df.zero[:365], c="k", linewidth=1, zorder=-1)
    plt.legend(
        handles=[
            Patch(facecolor=hex, edgecolor=hex, label=cluster)
            for hex, cluster in zip(
                ["#F50B00", "#F3700E", "#1ce1ce"],
                [
                    f"KDI (n={len(kdi_col)})",
                    f"Lac de Gras (LDG) (n = {len(ldg_col)})",
                    f"Yellowknife (YK) (n = {len(yk_col)})",
                ],
            )
        ],
        loc="upper right",
        fontsize="small",
    )
    plt.xticks(range(15, 375, 30), exp.szn_list)
    plt.xlabel("")
    plt.ylabel("Observed Temperature ˚C")

    if save:
        plt.savefig("/home/hma000/accomatic-web/plotting/out/clusters.png")


if sys.argv[1] == "-terr":
    exp = Experiment("/home/hma000/accomatic-web/data/toml/test.toml")
    terrain_timeseries(exp, save=True)


if sys.argv[1] == "-clus":
    exp = Experiment("/home/hma000/accomatic-web/data/toml/test.toml")
    cluster_timeseries(exp, save=True)
