from NcReader import *
import matplotlib.image as image
import matplotlib.pyplot as plt
import matplotlib.font_manager
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import random
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"


def heatmap_plot():
    df = pd.read_csv(
        "/home/hma000/accomatic-web/csvs/szn_stats.csv",
        index_col=[0, 1],
        names=["rmse", "mae", "r2"],
        header=0,
    )
    all = df.index.get_level_values(1).unique()
    newdf = df.loc[(df.index.get_level_values(1) == all[0])].mean()

    for setup in all[1:]:
        newdf = pd.concat(
            [newdf, df.loc[(df.index.get_level_values(1) == setup)].mean()], axis=1
        )

    newdf.columns = all
    month_to_season_dct = {
        "All-Time": range(1, 13),
        "Winter": [1, 2, 12],
        "Spring": [3, 4, 5],
        "Summer": [6, 7, 8],
        "Fall": [9, 10, 11],
    }

    all_time = newdf.filter(regex="All-Time").T
    all_time.rmse = all_time.rmse.rank(method="max")
    all_time.mae = all_time.rmse.rank(method="max")
    all_time.r2 = all_time.rmse.rank(method="max")

    winter = newdf.filter(regex="Winter").T
    winter.rmse = winter.rmse.rank(method="max")
    winter.mae = winter.rmse.rank(method="max")
    winter.r2 = winter.rmse.rank(method="max")

    spring = newdf.filter(regex="Spring").T
    spring.rmse = spring.rmse.rank(method="max")
    spring.mae = spring.rmse.rank(method="max")
    spring.r2 = spring.rmse.rank(method="max")

    summer = newdf.filter(regex="Summer").T
    summer.rmse = summer.rmse.rank(method="max")
    summer.mae = summer.rmse.rank(method="max")
    summer.r2 = summer.rmse.rank(method="max")

    fall = newdf.filter(regex="Fall").T
    fall.rmse = fall.rmse.rank(method="max")
    fall.mae = fall.rmse.rank(method="max")
    fall.r2 = fall.rmse.rank(method="max")

    sns.color_palette(["#1CE1CE", "#008080", "#F3700E"])
    sns.set_context("poster")
    sns.despine()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(25, 20))

    sns.heatmap(all_time, annot=True, ax=ax1, cmap="YlGnBu")
    sns.heatmap(winter, annot=True, ax=ax2, cmap="YlGnBu")
    sns.heatmap(summer, annot=True, ax=ax3, cmap="YlGnBu")
    sns.heatmap(spring, annot=True, ax=ax4, cmap="YlGnBu")
    sns.heatmap(fall, annot=True, ax=ax5, cmap="YlGnBu")
    plt.tight_layout()
    plt.savefig("heatmap_plot.png", dpi=300)


def allsites_timeseries_plot():
    pal = sns.dark_palette("#F3700E", 23)
    sns.set_context("poster")
    sns.despine()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})

    fig, ax = plt.subplots(figsize=(30, 13.5))

    sns.lineplot(
        x="time",
        y="soil_temperature",
        hue="sitename",
        data=odf,
        legend=False,
        palette=pal,
    )

    # Set title and labels for axes
    ax.set(xlabel="Date", ylabel="GST (C)")

    # Define the date format
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)

    plt.savefig("allsites_timeseries_plot.png", dpi=300, transparent=True)


def terraintype_timeseries_plot():
    df = pd.read_csv("terrains_df.csv")
    df.time = pd.to_datetime(df["time"], format="%Y-%m-%d")
    df = df[df.time.dt.month > 5]
    df = df[df.time.dt.month < 9]
    df = df.reset_index()
    df = df.set_index(["time", "simulation"])
    # Setting custom colour palette for seaborn plots
    palette = ["#1CE1CE", "#008080", "#F3700E", "#F50B00", "#59473C"]
    sns.set_palette(sns.color_palette(palette))
    sns.set_context("poster")
    sns.despine()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})

    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(
        x="time", y="soil_temperature", linewidth=4, data=df, ax=ax, hue="simulation"
    )

    # Set title and labels for axes
    ax.set(xlabel="Date", ylabel="GST (C˚)")

    ax.legend(["Clay", "Peat", "Sand", "Rock", "Gravel"], loc="best")

    # Define the date format
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)

    plt.savefig(
        "terraintype_timeseries_plot.png", dpi=300, legend=False, transparent=True
    )


def residual_timeseries_plot(exp):
    # This function needs work
    # But you're not supposed to be programming
    obs = read_nc(exp.obs_pth)
    mod = read_geotop(exp.model_pth)

    mod = mod.reset_index(drop=False)
    mod = mod.groupby("time").mean()
    obs = obs.reset_index(drop=False)
    obs = obs.groupby("time").mean()

    mod = mod.sub(obs["obs"], axis=0)
    mod = mod.dropna()
    mod = mod.resample("M").mean()
    plt.plot(mod)
    plt.savefig("residual_timeseries_plot.png")


def std_dev(exp):
    obs = read_nc(exp.obs_pth)
    mod = read_geotop(exp.model_pth)

    mod = mod.reset_index(drop=False)
    mod = mod.groupby("time").mean()
    obs = obs.reset_index(drop=False)
    obs = obs.groupby("time").mean()

    mod = mod.sub(obs["obs"], axis=0)
    mod = mod.dropna()

    df = mod.join(obs).dropna()
    print(df.std())


def residual(df, mode):
    mod = df.drop(["obs"], axis=1)
    if mode == "MAE":
        mod = mod.sub(df.obs, axis=0).abs()
    mod = mod.resample("W-MON").mean()
    return mod


def gaussian_smooth(x, y, grid, sd):
    weights = np.transpose([stats.norm.pdf(grid, m, sd) for m in x])
    weights = weights / weights.sum(0)
    return (weights * y).sum(1)


def streamline_plot(exp):
    obs, mod = read_nc(exp.obs_pth), read_geotop(exp.model_pth)
    obs, mod = obs.reset_index(drop=False), mod.reset_index(drop=False)
    obs, mod = obs.groupby("time").mean(), mod.groupby("time").mean()
    obs.index, mod.index = pd.to_datetime(obs.index), pd.to_datetime(mod.index)

    df = obs.join(mod).dropna()

    df = residual(df, "NONE")

    fig, ax = plt.subplots(figsize=(13, 7))

    x = df.index
    x_int = np.arange(1, len(x) + 1, dtype=int)
    y = df.to_numpy().T

    smooth_buffer = 3  # This makes the ends of the graph pointy

    grid = np.linspace(-smooth_buffer, x_int[-1] + smooth_buffer, num=500)
    y_smoothed = [gaussian_smooth(x_int, y_, grid, 0.85) for y_ in y]

    COLOURS = ["#1CE1CE", "#008080", "#F3700E"]
    ax.stackplot(grid, y_smoothed, colors=COLOURS)  # , baseline="sym")

    # x label shenanigans
    x_labels = x.to_period("Y").unique().to_list()
    upper = np.arange(0, len(grid), (len(grid) / len(x_labels)), dtype=int)
    ax.set_xticks([grid[i + smooth_buffer] for i in upper])
    ax.set_xticklabels(x_labels)

    # plt.axvspan(0, 100, facecolor='b', alpha=0.25, zorder=-100)
    # plt.axvspan(100, 200, facecolor='g', alpha=0.25, zorder=-100)

    ax.legend(df.columns)

    plt.ylabel("Modelled Temperature ˚C")
    plt.savefig("streamline_plot.png")


def xy_plot(exp):
    df = exp.results
    df = df.groupby(["sim", "site"]).mean()

    xlims, ylims = (3, 15), (3, 15)
    print(df.head())
    ax = sns.scatterplot(data=df, x="MAE", y="RMSE", hue="sim")
    ax.set(xlim=xlims, ylim=ylims)
    ax.plot(xlims, xlims, color="k")
    ax.set_aspect("equal", adjustable="box")

    plt.savefig("xy_plot.png")


def xy_stats_plot(exp):
    df = exp.obs("KDI-E-ShrubT")
    df = df.dropna().resample("W-MON").mean()

    df["Simulation (A)"] = df.obs + 3
    group = np.random.normal(loc=df.obs.mean(), scale=df.obs.std(), size=len(df.obs))
    df["Simulation (B)"] = (
        ((group - group.mean()) / group.std()) * df.obs.std()
    ) + df.obs.mean()
    df["Simulation (C)"] = df.obs * np.random.uniform(
        low=0.9, high=1.1, size=len(df.obs)
    )
    df["Simulation (C)"].iloc[np.random.choice(len(df.obs), replace=False, size=6)] = [
        10,
        -10,
        10,
        -10,
        10,
        -10,
    ]

    a = (
        df["Simulation (A)"].mean(),
        df["Simulation (A)"].std(),
        mean_squared_error(df["obs"], df["Simulation (A)"], squared=False),
    )
    b = (
        df["Simulation (B)"].mean(),
        df["Simulation (B)"].std(),
        mean_squared_error(df["obs"], df["Simulation (B)"], squared=False),
    )
    c = (
        df["Simulation (C)"].mean(),
        df["Simulation (C)"].std(),
        mean_squared_error(df["obs"], df["Simulation (C)"], squared=False),
    )

    r = "\n".join(
        (
            r"$Obs: \mu=%.2f , \sigma=%.2f$" % (df["obs"].mean(), df["obs"].std()),
            r"$(A): \mu=%.2f , \sigma=%.2f,  RMSE=%.2f$" % a,
            r"$(B): \mu=%.2f , \sigma=%.2f,  RMSE=%.2f$" % b,
            r"$(C): \mu=%.2f , \sigma=%.2f,  RMSE=%.2f$" % c,
        )
    )

    xlims, ylims = (df.obs.min(), df.obs.max()), (df.obs.min(), df.obs.max())
    df = df.reset_index(drop=False)
    df = df.melt(id_vars=["time", "obs"], var_name="sim", value_name="pred")
    df = df.set_index("time")
    palette = ["#1CE1CE", "#008080", "#F3700E", "#F50B00", "#59473C"]
    sns.set_palette(sns.color_palette(palette))
    plt.figure(figsize=(10, 10), facecolor="w")

    ax = sns.scatterplot(
        data=df,
        x="obs",
        y="pred",
        hue="sim",
        style="sim",
        legend=True,
    )

    ax.plot(xlims, xlims, color="k")
    ax.set(xlim=xlims, ylim=ylims)
    ax.set_aspect("equal", adjustable="box")

    props = dict(boxstyle="round", facecolor="white", alpha=1)
    plt.legend(bbox=props, fontsize=12, loc="center left")
    plt.text(
        0.05,
        0.95,
        r,
        fontsize=12,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=props,
    )
    plt.xlabel("Observations ˚C")
    plt.ylabel("Simulated Data ˚C")
    plt.savefig("xy_stats_plot.png")