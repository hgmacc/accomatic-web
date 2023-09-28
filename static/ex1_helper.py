import statistics as stats
import matplotlib.pyplot as plt
from static.statistics_helper import *
import time
import scipy
from dateutil import rrule
from matplotlib.backends.backend_pdf import PdfPages
from accomatic.Plotting import get_colour
from itertools import product


plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True


def bs_10_day_sample(exp, sample_size):
    o = exp.obs("ROCKT1")
    m = exp.mod("ROCKT1")["ens"]
    stat = "MAE"

    nrows = range(o.shape[0])
    res = []
    window = 10

    while len(res) < sample_size:
        ix = random.randint(nrows.start, nrows.stop - (window + 1))
        try:
            a = acco_measures[stat](o.iloc[ix : ix + window], m.iloc[ix : ix + window])
        except ValueError:
            continue
        res.append(a)
    return res


def get_ts_blocks(exp, missing_data=False):
    """
    If stat not specified, function uses RECURSION to run through all
    stats in exp.acco_list

    Reads in exp file paths
    Get's ts for: [each terrain x month]
    """

    res = {terr: [] for terr in exp.terr_list}
    merge = pd.concat([exp.obs(), exp.mod()], axis=1).dropna()
    merge["terrain"] = [
        exp.terr_dict()[site] for site in merge.index.get_level_values(1)
    ]

    for terr in set(exp.terr_list):
        # Average df over all sites for each day
        merge_t = (
            merge[merge["terrain"] == terr]
            .drop(columns=["terrain"])
            .groupby(level=[0])
            .mean()
        )
        merge_t.index = pd.DatetimeIndex(merge_t.index)
        for date in merge_t.index.to_period("m").unique():

            merge_i = merge_t.loc[(merge_t.index.to_period("m") == date)]

            if len(merge_i) < 27:
                # Bagging imputation here
                continue
            else:
                res[terr].append((merge_i))

    for i in res.keys():
        res[i] = pd.concat(res[i])
    res = pd.concat(res.values(), keys=res.keys())
    return res


def get_data(exp):
    """
    DATA [TERRAINS] [MONTH] : [LIST OF TS DATA FRAMES]
    """
    mod = exp.mod()
    obs = exp.obs()

    print(obs.head())
    sys.exit()

    return data


def get_acco_vals(exp, stat=""):
    """
    If stat not specified, function uses RECURSION to run through all
    stats in exp.acco_list

    Reads in exp file paths
    Get's accordance value for: [each terrain x month]
    """

    if stat == "":
        df_list = []
        for stat in exp.acco_list:
            df = get_acco_vals(exp, stat)
            df["stat"] = stat
            df_list.append(df)
        return pd.concat(df_list, axis=0).reset_index(drop=True)

    res = {mod: [] for mod in exp.mod_names()}
    res["data"] = []
    merge = pd.concat([exp.obs(), exp.mod()], axis=1).dropna()
    merge["terrain"] = [
        exp.terr_dict()[site] for site in merge.index.get_level_values(1)
    ]

    for terr in set(exp.terr_list):
        # Average df over all sites for each day
        merge_t = (
            merge[merge["terrain"] == terr]
            .drop(columns=["terrain"])
            .groupby(level=[0])
            .mean()
        )
        merge_t.index = pd.DatetimeIndex(merge_t.index)
        for i in range(1, 13):
            # Multi-year df; one month
            merge_i = merge_t.loc[(merge_t.index.month == i)]

            # If not a full month of data
            if len(merge_i) < 27:
                continue

            avg = []
            for model in exp.mod_names():
                avg = []
                for year in merge_i.index.year.unique():
                    merge_i_y = merge_i.loc[str(year)]
                    avg.append(acco_measures[stat](merge_i_y.obs, merge_i_y[model]))
                res[model].append((sum(avg) / len(avg)))

            res["data"].append(len(merge_i))

    return pd.DataFrame.from_dict(res)


def get_bs_data(sample, n=10000, missing_data=False):
    """
    Bootstraps acco_vals n (10,000) times
    Records mean of each bootstrap resample
    Mean accordance values: 4 Models x 10,000 Resamples
    """
    if not missing_data:
        missing_data = [0]

    bs_data = {proportion: {} for proportion in missing_data}

    # If not missing_data exp amt = [0] (nothing removed)
    for amt in list(bs_data.keys()):

        bs_mean = {mod: [] for mod in list(sample.keys())}
        bs_se = {mod: [] for mod in list(sample.keys())}

        # Generate random indices of size amt
        indices = np.random.choice(
            sample.index, size=int(amt / 100 * len(sample)), replace=False
        )

        # Remove rows at random
        sample_r = sample.drop(indices).reset_index(drop=True)

        for i in range(n):  # should only get MEANS from resample (1 value per model)
            bootstrap_sample = sample_r.iloc[
                random.choices(sample_r.index, k=len(sample_r))
            ]
            mean = bootstrap_sample.mean()
            se = bootstrap_sample.sem()
            for mod in list(bs_mean.keys()):
                bs_mean[mod].append(mean[mod])
                bs_se[mod].append(se[mod])

        bs_data[amt] = {
            "means": pd.DataFrame.from_dict(bs_mean),
            "se": pd.DataFrame.from_dict(bs_se),
        }

    return bs_data


def get_rank_distribution(x_star, stat):
    """
    Takes 4 models x 10,000 Resample means
    Ranks each instance
    Records proportion of time each model occupies each rank positon
    """
    # print("Running get_rank_distribution(df, stat).")
    if acco_rank[stat] == "max":
        x_star = x_star * -1
    ranks = x_star.rank(axis=1, method="max")
    tmp_list = []
    for mod in ranks.columns:
        tmp_ranks = ranks[mod].value_counts(normalize=True).sort_index().rename(mod)
        tmp_list.append(tmp_ranks)

    ranks = pd.concat(tmp_list, axis=1).transpose().fillna(0)
    ranks = ranks[[1.0, 2.0, 3.0, 4.0]]
    # ranks = ranks.sort_values(ranks.columns[0], ascending=False)
    return ranks


def terr_boxplot(data, stat):
    """
    Function takes and df with x 4 model columns
    Box and whisker plots results
    Will always plot in order: Best -> Worst
    """
    data = data[data.stat == "MAE"]
    data = data[["ens", "terr"]].pivot("terr")  # .to_numpy()
    print(data.head())
    fig, ax = plt.subplots(figsize=(5, 10))

    bp = ax.boxplot(data, whis=1.5, patch_artist=True, showmeans=True)

    ax.set_title(f"Bootstrapped rank results", fontsize="small")
    for patch, mod in zip(bp["boxes"], ci.columns):
        patch.set_facecolor(get_colour(mod))
    for mean in bp["means"]:
        mean.set_markerfacecolor("#000000")
        mean.set_markeredgecolor("#000000")
        mean.set_marker("D")
    for median in bp["medians"]:
        median.set_color("#000000")

    plt.title(stat)
    plt.savefig("/home/hma000/accomatic-web/plots/ex1/block_bs_boxplot.png")


def bs_boxplot(bs_data, sample, n, stat=""):
    """
    Function takes and df with x 4 model columns
    Box and whisker plots results
    Will always plot in order: Best -> Worst
    """

    ci_list = []
    for amt in list(bs_data.keys()):
        x_hat = bs_data[amt]["means"]
        # GET DATA
        x_hat = sample[sample.columns].mean()
        bs_var = bs_data[amt]["means"] - x_hat  # variation of bs resamples
        se = bs_data[amt]["se"]  # SE of each bs resample
        t_star = bs_var / se  # t_star = variations / SE
        t_star = pd.concat(
            [
                t_star[col].sort_values(ascending=False).reset_index(drop=True)
                for col in t_star
            ],
            axis=1,
            ignore_index=True,
        )
        t_star.columns = bs_var.columns.to_list()
        # se = t_star.sem().transpose()
        ci = x_hat - (t_star[int(0.025 * n) : int(0.975 * n)] * se)

        # Reorganize into first -> last
        order = ci.mean().sort_values().index.to_list()
        ci = ci[order]

        # Rename cols from era5 -> era5_1.0
        ci.columns = [f"{i}_{amt}" for i in ci.columns]
        ci_list.append(ci.dropna().reset_index(drop=True))

    if len(ci_list) > 1:
        ci = pd.concat(ci_list, axis=1)
        ci = ci.reindex(sorted(ci.columns), axis=1)
        mods = [i.split("_")[0] for i in list(ci.columns)][::-4]
        xticks = [i.split("_")[1] for i in list(ci.columns)]
        for i, mod in zip(range(len(ci.columns) - len(mods), -1, -len(mods)), mods):
            xticks.insert(i, f"\n\n{mod.upper()}")
    else:
        ci = ci_list[0]
        xticks = [i.split("_")[0].upper() for i in ci.columns]

    fig_box, ax = plt.subplots(figsize=(1.25 * len(ci.columns), 10))

    bp = ax.boxplot(ci, whis=1.5, patch_artist=True, showmeans=True)
    if len(ci_list) > 1:
        ax.set_xticks(
            [
                2.5,
                1,
                2,
                3,
                4,
                6.5,
                5,
                6,
                7,
                8,
                10.5,
                9,
                10,
                11,
                12,
                14.5,
                13,
                14,
                15,
                16,
            ]
        )
        ax.set_xticklabels(xticks)
        plt.xlabel("Percent of data removed.")
    else:
        ax.set_xticklabels(xticks)

    ax.set_title(f"Bootstrapped rank results n={n}", fontsize="small")
    for patch, mod in zip(bp["boxes"], ci.columns):
        patch.set_facecolor(get_colour(mod))
    for mean in bp["means"]:
        mean.set_markerfacecolor("#000000")
        mean.set_markeredgecolor("#000000")
        mean.set_marker("D")
    for median in bp["medians"]:
        median.set_color("#000000")

    plt.title(stat)
    plt.savefig("/home/hma000/accomatic-web/plots/ex1/bs_boxplot.png")


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


def bs_heatmap(ranks, amt=False):
    # print(f"Plotting bootstrap heatmap for n={len(ranks)}")

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
    if amt:
        plt.title(amt)
    # plt.savefig("/home/hma000/accomatic-web/plots/ex1/bs_SEA_heatmap.png")


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
