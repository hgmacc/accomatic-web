import pickle
import sys
import seaborn as sns
import matplotlib.patches as mpatches

sys.path.append("../")


import matplotlib.pyplot as plt
from accomatic.Experiment import *

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"


def terrain_pieplot(exp):
    exp = read_exp("/home/hma000/accomatic-web/data/pickles/09May_0.1_0.pickle")
    df = exp.obs().reset_index()
    df["terr"] = df.sitename.apply(lambda s: exp.terr_dict()[s])
    df = df.groupby(["terr", "sitename"]).count()["obs"].reset_index()
    import plotly.express as px

    terr_names = [
        "Peatland",
        "Coarse Hilltop",
        "Fine Hilltop",
        "Snowdrift",
        "Horizontal Rock",
    ]
    df["terr"] = df.terr.apply(lambda t: terr_names[t - 1])
    df.head()

    fig = px.sunburst(
        df,
        path=["terr", "sitename"],
        values="obs",
        color="terr",
        color_discrete_map=dict(
            zip(terr_names, ["#FF7F50", "#00CED1", "#FFD700", "#4682B4", "#778899"])
        ),
    )

    fig.update_layout(
        width=600,
        height=600,
    )
    fig.write_image("figure.png", scale=2)


def terrain_timeseries(exp, mod=False, save=False):
    ######## TERRAIN PLOT #######################
    fig, axs = plt.subplots(5, 1, sharey=True, sharex=True, figsize=(13, 15))

    o = exp.obs().reset_index(drop=False)

    o.level_0 = pd.to_datetime(o.level_0)
    o["day-month"] = o.level_0.dt.strftime("%m-%d")
    o["year"] = o.level_0.dt.strftime("%y")
    o.set_index("day-month", inplace=True)
    o = o.drop(columns="level_0")
    # pretty sure this next commented out line would
    # merge all the sites into one sns lineplot spread
    # o = o.groupby(["day-month", "sitename"]).mean().drop(columns="level_0")
    o["terr"] = [exp.terr_dict()[x] for x in o.sitename]
    o["sitename"] = o.sitename.str[:2]
    o.loc[o.sitename.isin(["RO", "NG"]), 'sitename'] = "LDG"
    o.loc[o.sitename == "KD", 'sitename'] = "KDI"
    terr_desc = [
        "Peatland",
        "Coarse Hilltop",
        "Fine Hilltop",
        "Snowdrift",
        "Horizontal Rock",
    ]
    sitename_colors = {"KDI":"#F50B00", "YK":"#F3700E", "LDG":"#1ce1ce"}
    sns.set_palette(sns.color_palette([sitename_colors.get(k, 'grey') for k in sorted(sitename_colors.keys())]))
    terr_dict = dict(zip(range(1, 7), terr_desc))
    o.terr = [terr_dict[i] for i in o.terr]
    for i in terr_dict.keys():
        plt.subplot(5, 1, i)
        sns.lineplot(
            data=o[o.terr == terr_dict[i]].dropna(),
            x="day-month",
            y="obs",
            hue="sitename",
            hue_order=["KDI","YK","LDG"],
            palette=sitename_colors,
            errorbar=("sd", 1),
            legend=False,
        )
        if i == 1:
            legend_handles = [mpatches.Patch(color=color, label=label) for label, color in sitename_colors.items()]
            plt.legend(handles=legend_handles, loc='lower left', frameon=False)
            
        plt.axhline(y=0, color="k", linestyle="-", linewidth=0.5, zorder=-1)
        # l = plt.legend(
        #     title="",
        #     loc="upper left",
        #     fontsize="xx-small",
        # )
        if i == 5:
            months = ["JAN", "MAR", "MAY", "JUL", "SEP", "NOV"]
            plt.xticks(ticks=range(1, 365, 62), labels=months)
        plt.xlabel("")
        plt.ylabel("")
    fig.supylabel("Observed Temperature ˚C")
    plt.tight_layout()
    if save:
        plt.savefig("/home/hma000/accomatic-web/plotting/out/examples/terrains.png")


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
        errorbar=("sd", 1),
        palette=["#F50B00", "#F3700E", "#1ce1ce"],
        legend=False,
    )
    # plotting 0˚C line
    df["zero"] = 0
    plt.plot(df.zero[:365], c="k", linewidth=1, zorder=-1)
    plt.legend(
        handles=[
            mpatches.Patch(facecolor=hex, edgecolor=hex, label=cluster)
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
        plt.savefig("/home/hma000/accomatic-web/plotting/out/examples/clusters.png")


if sys.argv[1] == "-terr":
    exp = Experiment("/home/hma000/accomatic-web/data/toml/run.toml")
    terrain_timeseries(exp, mod=True, save=True)


if sys.argv[1] == "-clus":
    exp = Experiment("/home/hma000/accomatic-web/data/toml/test.toml")
    cluster_timeseries(exp, save=True)
