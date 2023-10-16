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


def boxplot_allterrains(exp, stat):

    # Create each terrain - each has varying amounts of data for sure
    # Just use ens output
    res = exp.results.loc["res"]
    idx = pd.IndexSlice
    terrains = {}
    terr_desc = [
        "Eskers",
        "Wet Sedge",
        "Rock",
        "Polygons & Shrubs",
        "Shrubby Hummuck",
    ]
    sum = 0
    for terrain in set(exp.terr_list):
        for month in exp.data[terrain].keys():
            sum = sum + len(exp.data[terrain][month])

        arr = np.sort(
            np.concatenate(
                [i.arr for i in res.loc[idx[terrain, :, stat], idx["ens"]].tolist()]
            )
        )
        terrains[f"{terr_desc[terrain - 1]} n=({sum})"] = arr[
            int(0.025 * len(arr)) : int(0.975 * len(arr))
        ]
    data = pd.DataFrame.from_dict(terrains)
    bs_boxplot(data, title=f"{stat} across all terrains.")


def boxplot_alldata():
    data = {}
    for mod in exp.mod_names():
        arr = np.sort(
            np.concatenate([i.arr for i in res.loc[idx[:, :, stat], idx[mod]].tolist()])
        )
        data[mod] = arr[int(0.025 * len(arr)) : int(0.975 * len(arr))]
    data = pd.DataFrame.from_dict(data)
    bs_boxplot(exp, stat)


def missing_boxplot(explist):
    data = {}
    # data = {'0': array, '25': array, '50': array}
    idx = pd.IndexSlice
    for exp in explist:
        res = exp.results.loc["res"]
        arr = np.sort(
            np.concatenate(
                [i.arr for i in res.loc[idx[:, :, "MAE"], idx["ens"]].tolist()]
            )
        )
        data[str(exp.missing_data)] = arr[int(0.025 * len(arr)) : int(0.975 * len(arr))]
    data = pd.DataFrame.from_dict(data)
    print(data.head())
    bs_boxplot(data, "bias_missing")


def build_missing_data_pickles(n, missing_amts):
    # missing_amts = [list, of, missing, data, percents]
    for amt in missing_amts:
        exp = Experiment("tests/test_data/toml/ykl.toml")
        exp.missing_data = amt
        exp.boot_size = n
        build(exp)
        concatonate(exp)
        with open(
            f"tests/test_data/pickles/exp_{exp.missing_data}.pickle",
            "wb",
        ) as handle:
            pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_pickles(pickle_dir):
    exp_list = []
    for filename in os.listdir(pickle_dir):
        f = os.path.join(pickle_dir, filename)
        if os.path.isfile(f):
            with open(f, "rb") as handle:
                exp_list.append(pickle.load(handle))

    return exp_list


if __name__ == "__main__":
    s = time.time()

    build_missing_data_pickles(n=1000, missing_amts=[0, 10, 25])
    # missing_boxplot(get_pickles("tests/test_data/pickles/n_10"))

    print(f"This took {format(time.time() - s, '.2f')}s to run.")
