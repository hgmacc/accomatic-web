import statistics as stats
import matplotlib.pyplot as plt
from static.statistics_helper import *
import time
import scipy
from dateutil import rrule
from matplotlib.backends.backend_pdf import PdfPages
from accomatic.Plotting import get_colour
from itertools import product
from matplotlib.patches import Patch


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


def get_bs_data(n, sample):
    """
    Bootstraps acco_vals n (10,000) times
    Records mean of each bootstrap resample
    Mean accordance values: 4 Models x 10,000 Resamples
    """
    bs_mean = {mod: [] for mod in list(sample.keys())}
    bs_se = {mod: [] for mod in list(sample.keys())}

    for i in range(n):  # should only get MEANS from resample (1 value per model)
        bootstrap_sample = sample.iloc[random.choices(sample.index, k=len(sample))]
        mean = bootstrap_sample.mean()
        se = bootstrap_sample.sem()
        for mod in list(bs_mean.keys()):
            bs_mean[mod].append(mean[mod])
            bs_se[mod].append(se[mod])

    bs_data = {
        "means": pd.DataFrame.from_dict(bs_mean),
        "se": pd.DataFrame.from_dict(bs_se),
    }
    return bs_data


def get_rank_distribution(exp, sett="all"):
    """
    sett: Dictionary {'terr' : terrains to incorporate, 'szn': seasons to aggregate}
    Records proportion of time each model occupies each rank positon

    exp.results = [terr][szn][ranks]
    """
    # if sett != "all":
    #     terrains = sett["terr"]
    ranks = exp.results
    tmp_list = []
    for mod in ranks.columns:
        tmp_ranks = ranks[mod].value_counts(normalize=True).sort_index().rename(mod)
        tmp_list.append(tmp_ranks)

    ranks = pd.concat(tmp_list, axis=1).transpose().fillna(0)
    ranks = ranks[[1.0, 2.0, 3.0, 4.0]]
    return ranks


PALLETTE = ["#527206", "#584538", "#008184", "#F50400", "#15e2d0"]


def bs_boxplot(data, title):
    """
    Function takes and df with x columns
    If you want colour coordination with models,
    column name must start with "mod_"

    i.e. "era5_jan_rock" -> c="#1ce1ce"

    Box and whisker plots results
    Will always plot in order: Best -> Worst
    """
    # data = data.iloc[:, [3, 4, 0, 1, 2]]
    fig_box, ax = plt.subplots(figsize=(1.25 * len(data.columns), 10))

    bp = ax.boxplot(data, whis=1.5, sym="", patch_artist=True, showmeans=True)

    # xticks = data.columns
    xticks = ["" for i in data.columns]
    ax.set_xticklabels(xticks, fontsize=12, rotation=90)
    ax.set_title(f"Bootstrapped rank results n={len(data)}", fontsize="small")

    for patch, col in zip(bp["boxes"], data.columns):
        print(col)
        if get_colour(col):
            patch.set_facecolor(get_colour(col))
        else:
            patch.set_facecolor(PALLETTE[data.columns.get_loc(col)])
    for mean in bp["means"]:
        mean.set_markerfacecolor("#000000")
        mean.set_markeredgecolor("#000000")
        mean.set_marker("D")

    for median in bp["medians"]:
        median.set_color("#000000")

    legend_elements = []
    for c in data.columns:
        if get_colour(c):
            legend_elements.append(
                Patch(facecolor=get_colour(c), edgecolor=get_colour(c), label=c)
            )
        else:
            legend_elements.append(
                Patch(
                    facecolor=PALLETTE[data.columns.get_loc(c)],
                    edgecolor=PALLETTE[data.columns.get_loc(c)],
                    label=c,
                )
            )

    plt.legend(handles=legend_elements)
    plt.title(title)
    plt.savefig(f"/home/hma000/accomatic-web/plots/ex1/bs_boxplot_{title}.png")


def bs_histogram(n, sample, bs_data):
    """
    Plots histogram for bootstrap and t-interval bootstrap
    """
    # print(f"Plotting bootstrap histogram for n={len(bs_data['means'])}")

    # GET DATA
    x_star = bs_data["means"]
    x_hat = sample.mean()
    bs_var = x_star - x_hat
    se = bs_data["se"]  # SE of each bs resample
    t_star = bs_var / se

    # REGULAR BOOTSTRAP
    bs_var = pd.concat(
        [
            bs_var[col].sort_values(ascending=False).reset_index(drop=True)
            for col in x_star
        ],
        axis=1,
        ignore_index=True,
    )
    bs_var.columns = sample.columns.to_list()
    bs_var = bs_var[int(0.025 * n) : int(0.975 * n)]

    # WEIRD T-INTERVAL BOOTSTRAP
    t_star = pd.concat(
        [
            t_star[col].sort_values(ascending=False).reset_index(drop=True)
            for col in t_star
        ],
        axis=1,
        ignore_index=True,
    )
    t_star.columns = sample.columns.to_list()
    ci = t_star[int(0.025 * n) : int(0.975 * n)] * se

    # PLOTTING
    fig_hist, (ax_t, ax_x) = plt.subplots(
        nrows=2, ncols=1, sharex=True, figsize=(6, 10)
    )
    for i, mod in zip(range(len(bs_var.columns)), bs_var.columns):
        ax_x.hist(bs_var[mod], histtype="step", color=get_colour(mod), label=mod)
        ax_t.hist(
            ci[mod],
            histtype="step",
            linestyle="dashed",
            color=get_colour(mod),
            label=f"{mod}_t",
        )

    ax_x.legend()
    ax_x.set_title("Regular bootstrap")
    ax_t.set_title("t bootstrap")


def bs_heatmap(ranks, title=""):
    x_ranks = ["First", "Second", "Third", "Fourth"]
    y_mod = [x.upper() for x in ranks.index]

    data = ranks.to_numpy()

    fig_heat, ax = plt.subplots(figsize=(7, 9))

    sns.heatmap(
        data,
        vmin=0,
        vmax=1,
        cmap="YlOrRd",
        annot=True,
        fmt=".2g",
        linewidths=1,
        linecolor="white",
        cbar=True,
        cbar_kws={
            "label": "Proportion of instances occupying rank",
            "shrink": 0.75,
            "location": "bottom",
            "pad": 0.01,
            "ticks": [0.0, 0.5, 1.0],
        },
        square=True,
        xticklabels=x_ranks,
        yticklabels=y_mod,
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    plt.savefig(f"/home/hma000/accomatic-web/plots/ex1/bs_{title}.png")


def bias_heatmap(x_star):
    x_ranks = ["Bias"]
    y_mod = [x.upper() for x in x_star.columns]

    data = np.array([[i] for i in (x_star[x_star > 0].count() / len(x_star))])

    fig_bias, ax = plt.subplots(figsize=(5, 8))
    plt.rcParams["font.size"] = "12"

    sns.heatmap(
        data,
        vmin=0,
        vmax=1,
        cmap="RdBu_r",
        annot=True,
        fmt=".2g",
        linewidths=1,
        linecolor="white",
        cbar=True,
        cbar_kws={
            "label": "Proportion of positive bias",
            "ticks": [0.0, 0.5, 1.0],
        },
        square=True,
        xticklabels=x_ranks,
        yticklabels=y_mod,
    )
    ax.xaxis.tick_top()
    ax.tick_params(length=0)

    plt.title("Bias distribution of bias bootstrap results")
    plt.savefig("/home/hma000/accomatic-web/plots/ex1/bs_bias_heatmap.png")


def save_image(filename):
    p = PdfPages(filename)

    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    for fig in figs:
        fig.savefig(p, format="pdf")
    p.close()
    plt.close("all")
