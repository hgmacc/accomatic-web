from Experiment import *
from Plotting import *
from static.ex1_helper import *
from itertools import product
from Stats import *

import pickle


def run_ex1(sample, n, stat, missing_data=False):

    bs_data = get_bs_data(
        sample=sample[sample.stat == stat]
        .drop(columns=["stat", "data"])
        .reset_index(drop=True),
        n=n,
        missing_data=missing_data,
    )

    bs_boxplot(
        bs_data=bs_data,
        sample=sample[sample.stat == stat].drop(columns=["stat", "data"]),
        n=n,
    )

    rank_dist = get_rank_distribution(x_star=bs_data[0]["means"], stat=stat)

    bs_histogram(
        n=n,
        sample=sample[sample.stat == stat].drop(columns=["stat", "data"]),
        bs_data=bs_data[0],
    )
    if stat == "BIAS":
        bias_heatmap(x_star=bs_data[0]["means"])
    else:
        bs_heatmap(rank_dist)
    amt = ""
    if missing_data:
        amt = f"_{missing_data[0]}"
    save_image(f"plots/ex1/multi_plot_{stat}{amt}.pdf")


def nested_boot(sample, n, stat, missing_data):
    x_star = {amt: [] for amt in missing_data}
    se_star = {amt: [] for amt in missing_data}
    for i in range(100):

        # Get bs old_results after removing data
        bs_data = get_bs_data(
            sample=sample[sample.stat == stat]
            .drop(columns=["stat", "data"])
            .reset_index(drop=True),
            n=n,
            missing_data=missing_data,
        )

        # For each amt, save the mean values of each model
        for amt in missing_data:
            x_star[amt].append(bs_data[amt]["means"].mean())
            se_star[amt].append(bs_data[amt]["se"].mean())

    nested_bs_data = {proportion: {} for proportion in missing_data}
    for amt in missing_data:
        nested_bs_data[amt] = {
            "means": pd.concat(x_star[amt], axis=1).transpose(),
            "se": pd.concat(se_star[amt], axis=1).transpose(),
        }

    bs_boxplot(
        bs_data=nested_bs_data,
        sample=sample[sample.stat == stat].drop(columns=["stat", "data"]),
        n=n,
        stat=stat,
    )


if __name__ == "__main__":
    s = time.time()

    # Load data (deserialize)
    with open("exp_100.pickle", "rb") as handle:
        exp = pickle.load(handle)

    print(exp.results.loc["rank"].head())

    print(f"This took {format(time.time() - s, '.2f')}s to run.")
