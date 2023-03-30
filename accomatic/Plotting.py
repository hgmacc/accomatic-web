import random

import matplotlib.font_manager
import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from accomatic.NcReader import *
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "10"

PLOT_PTH = '/home/hma000/accomatic-web/tests/plots/JAN31/'

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

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
    plt.savefig(f'{PLOT_PTH}heatmap_plot.png', dpi=300)


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.gcf().axes[1].invert_yaxis()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), ha="center",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def allsites_timeseries_plot(odf):
    pal = sns.dark_palette("#F3700E", 23)
    sns.set_context("poster")
    sns.despine()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})

    fig, ax = plt.subplots(figsize=(30, 13.5))

    sns.lineplot(
        x="time",
        y="obs",
        hue="sitename",
        data=odf,
        legend=False,
        #palette=pal,
    )

    # Set title and labels for axes
    ax.set(xlabel="Date", ylabel="GST (C)")

    # Define the date format
    from matplotlib.dates import DateFormatter
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)

    plt.savefig(f'{PLOT_PTH}allsites_timeseries_plot.png', dpi=300, transparent=True)


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
    from matplotlib.dates import DateFormatter
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)

    plt.savefig(
        f'{PLOT_PTH}terraintype_timeseries_plot.png', dpi=300, legend=False, transparent=True
    )


def residual_timeseries_plot(obs, mod):
    mod = mod.reset_index(drop=False)
    mod = mod.groupby("time").mean()
    obs = obs.reset_index(drop=False)
    obs = obs.groupby("time").mean()

    mod = mod.sub(obs["obs"], axis=0)
    mod = mod.dropna()
    mod = mod.resample("M").mean()
    plt.plot(mod)
    plt.savefig(f'{PLOT_PTH}residual_timeseries_plot.png')


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
    plt.savefig(f'{PLOT_PTH}streamline_plot.png')


def xy_plot(exp):
    df = exp.results
    df = df.groupby(["sim", "terr"]).mean()

    ax = sns.scatterplot(data=df, x="MAE", y="RMSE", hue="sim")
    ax.set(xlim=xlims, ylim=ylims)
    ax.plot(xlims, xlims, color="k")
    ax.set_aspect("equal", adjustable="box")

    plt.savefig(f'{PLOT_PTH}xy_plot.png')


def xy_stats_plot(exp):
    df = exp.obs()
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
    plt.savefig(f'{PLOT_PTH}xy_stats_plot.png')

    
def terr_timeseries_plot(exp, terr):
    df = exp.obs()
    ncol = len(df.index.get_level_values(1).unique())
    terr_dict = exp.terr_dict()
    terr_list = []
    for i in df.index.get_level_values(1):
        try: terr_list.append(terr_dict[i])
        except KeyError:
            terr_list.append(-1)
    
    df['terrain'] = terr_list
    df = df[df.terrain == terr].drop(["terrain"], axis=1)

    pal = sns.dark_palette("#F3700E", ncol)
    sns.set_context("poster")
    sns.despine()
    sns.set_style({"font.family": "serif", "font.serif": "Times New Roman"})

    fig, ax = plt.subplots(figsize=(30, 13.5))

    sns.lineplot(
        x="time",
        y="obs",
        hue="sitename",
        data=df,
        legend=False,
    )

    # Set title and labels for axes
    ax.set(xlabel="Date", ylabel="GST (C)")

    # Define the date format
    from matplotlib.dates import DateFormatter
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)
    plt.title(f"Terrain No. {terr}")
    plt.savefig(f'{PLOT_PTH}terrain_no{terr}_plot.png', dpi=300)#, transparent=True)


def xy_site_plot(exp,site):
    pth = '/home/hma000/accomatic-web/tests/plots/xy_plots/tmp/'
    odf = exp.obs(site)
    mdf = exp.mod(site)
    df = odf.join(mdf).dropna()
    df = df.resample("W-MON").mean()
    df['ens'] = df[['era5', 'jra55', 'merra2']].mean(axis=1)

    a, b = df.to_numpy().min(), df.to_numpy().max()
    lims = [(a, a),(b,b)]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=False, figsize=(10,13), squeeze=True)
    
    palette = ["#008080", "#F50B00", "#F3700E", "#59473c"]

    # XY SCATTER PLOT
    plt.subplot(211)
    ax1.set_aspect("equal")
    plt.scatter(df.obs, df.jra55, s=5, c=palette[0], label=f'JRA55 r={np.corrcoef(df.obs, df.jra55)[0][1]:.2f}')
    plt.scatter(df.obs, df.era5, s=5, c=palette[1], label=f'ERA5 r={np.corrcoef(df.obs, df.era5)[0][1]:.2f}')
    plt.scatter(df.obs, df.merra2, s=5, c=palette[2], label=f'MERRA2 r={np.corrcoef(df.obs, df.merra2)[0][1]:.2f}')
    plt.scatter(df.obs, df.ens, s=5, c=palette[3], label=f'ENSEMBLE r={np.corrcoef(df.obs, df.ens)[0][1]:.2f}')
    plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    plt.xlabel("Observed")
    plt.title(f"{site}")
    plt.ylabel("GEOtop")
    plt.legend(fontsize='x-small')
    plt.ylim((a,b))
    plt.xlim((a,b))

    # TIME SERIES 
    plt.subplot(212)
    plt.plot(df['obs'], c='k', label='obs',linewidth=2)
    for col, c in zip(df.drop(["obs"], axis=1).columns, palette):
        plt.plot(df[col], c=c, label=col)
        
    #ax2.set_aspect(23)
    plt.xlabel("Time")
    plt.ylabel("Temperature ˚C")
    plt.legend(fontsize='x-small')
    plt.xticks(rotation=70)
    fig.savefig(f'{pth}xy_{site}_plot.png')
    fig.clf()
    plt.close(fig)

