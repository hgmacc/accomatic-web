from Experiment import *
from Plotting import *
import statistics as stats
import matplotlib.pyplot as plt
from static.statistics_helper import *
import time
import scipy
from dateutil import rrule
from matplotlib.backends.backend_pdf import PdfPages

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


def get_acco_vals(exp, stat):
    """
    If stat not specified, function uses RECURSION to run through all
    stats in exp.acco_list

    Reads in exp file paths
    Get's accordance value for: [each site x each full month]
    """

    if stat == "":
        df_list = []
        for stat in exp.acco_list:
            df = get_acco_vals(exp, stat)
            df["stat"] = stat
            df_list.append(df)
        return pd.concat(df_list)

    res = {mod: [] for mod in exp.mod_names()}

    for site in exp.sites_list:
        merge = exp.obs(site).join(exp.mod(site)).dropna()
        months = merge.obs.resample("M").count()
        months = months[months > 27].index.strftime("%Y-%m").to_list()

        for i in months:
            merge_i = merge.loc[i]
            for model in res.keys():
                res[model].append(acco_measures[stat](merge_i.obs, merge_i[model]))

    return pd.DataFrame.from_dict(res)


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


def get_rank_distribution(x_star, stat):
    """
    Takes 4 models x 10,000 Resample means
    Ranks each instance
    Records proportion of time each model occupies each rank positon
    """
    print("Running get_rank_distribution(df, stat).")

    # TODO Fix weird danking; it won't rank by max
    x_star = x_star * -1  # Temp solution
    ranks = x_star.rank(axis=1, method="max")

    tmp_list = []
    for mod in ranks.columns:
        tmp_list.append(ranks[mod].value_counts(normalize=True).sort_index())
    ranks = pd.concat(tmp_list, axis=1).transpose().fillna(0)
    ranks = ranks[[1.0, 2.0, 3.0, 4.0]]
    ranks = ranks.sort_values(ranks.columns[0], ascending=False)

    return ranks


def bs_boxplot(bs_data, sample):
    """
    Function takes and df with x 4 model columns
    Box and whisker plots results
    Will always plot in order: Best -> Worst
    """
    print(f"Plotting bootstrap boxplot for n={len(bs_data['means'])}")

    # GET DATA
    x_hat = sample[sample.columns].mean()
    bs_var = bs_data["means"] - x_hat  # variation of bs resamples
    se = bs_data["se"]  # SE of each bs resample
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

    ci = x_hat - (t_star[int(0.025 * n) : int(0.975 * n)] * se)

    order = ci.mean().sort_values().index.to_list()
    ci = ci[order].dropna()

    fig_box, ax = plt.subplots(figsize=(6, 10))
    bp = ax.boxplot(ci, whis=1.5, patch_artist=True)
    ax.set_xticklabels(ci.columns)
    ax.set_title(f"Bootstrapped rank results n={len(ci)}")
    for patch, mod in zip(bp["boxes"], ci.columns):
        patch.set_facecolor(get_colour(mod))
    for median in bp["medians"]:
        median.set_color("#59473C")


def bs_histogram(n, sample, bs_data):
    """
    Plots histogram for bootstrap and t-interval bootstrap
    """
    print(f"Plotting bootstrap histogram for n={len(bs_data['means'])}")

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


def bs_heatmap(ranks):
    print(f"Plotting bootstrap heatmap for n={len(ranks)}")
    x_ranks = ["First", "Second", "Third", "Fourth"]

    y_mod = ranks.index

    data = ranks.to_numpy()

    fig_heat, ax = plt.subplots(figsize=(7, 9))
    plt.rcParams["font.size"] = "8"
    im, cbar = heatmap(
        data,
        y_mod,
        x_ranks,
        ax=ax,
        cmap="YlOrRd",
        cbarlabel="Proportion of instances occupying rank.",
    )

    texts = annotate_heatmap(im, data=data, valfmt="{x:.2f}")


def bias_heatmap(x_star):
    x_ranks = ["Bias"]
    y_mod = x_star.columns

    x_star = x_star[x_star > 0].count() / len(x_star)

    data = x_star.to_numpy()

    fig_bias, ax = plt.subplots(figsize=(8, 12))
    plt.rcParams["font.size"] = "12"
    im, cbar = heatmap(
        data,
        y_mod,
        x_ranks,
        ax=ax,
        cmap="RdBu",
        cbarlabel="Proportion of instances occupying +/- bias.",
    )
    texts = annotate_heatmap(im, data=data, valfmt="{x:.2f}")
    plt.title("Bias distribution of bias bootstrap results")
    plt.savefig("/home/hma000/accomatic-web/plots/ex1/bs_bias_heatmap.png")


def save_image(filename):
    p = PdfPages(filename)

    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]

    for fig in figs:
        fig.savefig(p, format="pdf")
    p.close()


if __name__ == "__main__":
    s = time.time()
    n = 100
    stat = "BIAS"

    # sample = get_acco_vals(exp)

    sample = pd.read_csv("tmp_ex1.csv")
    bs_data = get_bs_data(
        n=n,
        sample=sample[sample.stat == stat]
        .drop(columns=["stat"])
        .reset_index(drop=True),
    )

    bias_heatmap(x_star=bs_data["means"])
    sys.exit()

    rank_dist = get_rank_distribution(x_star=bs_data["means"], stat=stat)

    bs_boxplot(
        bs_data=bs_data, sample=sample[sample.stat == stat].drop(columns=["stat"])
    )
    bs_histogram(
        n=n, sample=sample[sample.stat == stat].drop(columns=["stat"]), bs_data=bs_data
    )
    if stat == "BIAS":
        bias_heatmap(x_star=bs_data["means"])
    else:
        bs_heatmap(rank_dist)

    save_image(f"plots/ex1/{stat}_multi_plot.pdf")

    print(f"This took {time.time() - s}s to run.")
