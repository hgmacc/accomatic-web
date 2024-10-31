from operator import indexOf
import random
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from accomatic.NcReader import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
import time
import pickle
from datetime import date
from typing import List
import math


class Cell:
    _arr: np.array

    def __init__(self):
        self._arr = []

    @property
    def arr(self) -> np.array:
        return np.array(self._arr)

    @arr.setter
    def arr(self, arr) -> None:
        self._arr = arr

    def ap(self, value: float) -> None:
        self._arr.append(value)

    def ranks(self) -> List[int]:
        a = list(self._arr)
        ranks = [a.count(rank) for rank in range(1, 5)]
        return ranks

    def bias(self) -> List[int]:
        # WARM BIAS
        bias = sum(i > 0 for i in list(self._arr))
        return bias

    def __repr__(self):
        return repr(self.arr)


class Data:
    _v: np.array

    def __init__(self, v):
        if type(v) == str:
            v = [int(i) for i in v.strip("[]").split(",")]
            v = np.array(v)
            rand_indices = np.random.randint(low=0, high=999, size=300)
            v[rand_indices] = np.nan
        self._v = v

    @property
    def v(self) -> np.array:
        return self._v

    @property
    def mean(self) -> float:
        return np.mean(self._v)

    @property
    def p(self) -> np.array:
        spl = make_interp_spline(range(10), self.v, k=3)
        return spl(np.linspace(-1, 1, 300))

    def __repr__(self):
        return repr(list(self.v))


def average_data(df_col):
    # df_col: column of np.arrays
    arr = np.array([i.arr for i in df_col.to_list()])
    return np.nanmean(arr.flatten("F")).round(3)


def rank_shifting_for_heatmap(df_col):
    return len(df_col.unique())


def std_dev(mod_ensemble):
    # From Luis (2020)
    M = len(mod_ensemble.columns)
    all_x_bars = []
    for model in mod_ensemble:
        all_x_bars.append(mod_ensemble[model].mean())
    x_bars_mean = np.mean(all_x_bars)
    return math.sqrt((((all_x_bars - x_bars_mean) ** 2).sum()) / (M - 1))


def variance(mod_ensemble):
    # From Luis (2020)
    M = len(mod_ensemble.columns)
    all_x_bars = []
    for model in mod_ensemble:
        all_x_bars.append(mod_ensemble[model].mean())
    x_bars_mean = np.mean(all_x_bars)
    return (((all_x_bars - x_bars_mean) ** 2).sum()) / (M - 1)


def d(p, o):
    o_mean = np.mean(o)
    sq_err = sum((p - o) ** 2)
    sq_dev = sum((abs(p - o_mean) + abs(o - o_mean)) ** 2)
    return 1 - (sq_err / sq_dev)


def d_1(p, o):
    o_mean = np.mean(o)
    abs_err = sum(abs(p - o))
    abs_dev = sum((abs(p - o_mean) + abs(o - o_mean)))
    return 1 - (abs_err / abs_dev)


def d_r(p, o):
    o_mean = np.mean(o)
    o_dev_x2 = 2 * sum(abs(o - o_mean))
    abs_err = sum(abs(p - o))
    if abs_err <= o_dev_x2:
        return 1 - (abs_err / o_dev_x2)
    if abs_err > o_dev_x2:
        return (o_dev_x2 / abs_err) - 1


def r_score(p, o):
    p_mean = np.mean(p)
    o_mean = np.mean(o)

    numer = sum((p - p_mean) * (o - o_mean))
    denom = sum((p - p_mean) ** 2) * sum((o - o_mean) ** 2)
    res = numer / math.sqrt(denom)
    return res


def nse_one(p, o):
    o_mean = o.mean()
    abs_err = sum(abs(o - p))
    o_dev = sum(abs(o - o_mean))
    return 1 - (abs_err / o_dev)


def bias(p, o):
    return np.mean(p - o)


def rmse(p, o):
    return mean_squared_error(o, p, squared=False)


stat_measures = {
    "RMSE": rmse,
    "R": r_score,
    "E1": nse_one,
    "MAE": mean_absolute_error,
    "d": d,
    "d1": d_1,
    "dr": d_r,
    "BIAS": bias,
}

stat_rank = {
    "RMSE": "min",
    "R": "max",
    "E1": "max",
    "MAE": "min",
    "d": "max",
    "d1": "max",
    "dr": "max",
    "BIAS": "min",
}

time_code_months = {
    "ALL": list(range(1, 13)),
    "DJF": [1, 2, 12],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "FREEZE": [10, 11, 12, 1, 2, 3],
    "THAW": [4, 5, 6, 7, 8, 9],
    "JAN": [1],
    "FEB": [2],
    "MAR": [3],
    "APR": [4],
    "MAY": [5],
    "JUN": [6],
    "JUL": [7],
    "AUG": [8],
    "SEP": [9],
    "OCT": [10],
    "NOV": [11],
    "DEC": [12],
}



def evaluate(exp, block):

    stat_dict = {
        stat: {"res": pd.DataFrame, "rank": pd.DataFrame} for stat in exp.stat_list
    }
    for stat in exp.stat_list:
        res = {mod: [] for mod in exp.mod_names()}
        # {'era5': 0.7, 'jra55': 0.98,  'merra': 0.25,  'ens': 0.10}
        # {'era5': 3,   'jra55': 4,     'merra': 2,     'ens': 0.4}
        for model in exp.mod_names():
            res[model].append(stat_measures[stat](block[model], block.obs))
        res = pd.DataFrame.from_dict(res)

        if stat == "BIAS" or stat == "MAE":
            rank = res.abs().rank(method="min", axis=1)
        if stat == "d" or stat == "R":
            res = res * -1
            rank = res.rank(method="max", axis=1)
            res = res * -1

        stat_dict[stat]["res"] = res
        stat_dict[stat]["rank"] = rank

    return stat_dict


def build(exp):
    print(f"Building {exp.boot_size} bootstrap; {exp.missing_data} missing data ...")
    s = time.time()
    for terr in exp.data.keys():
        for szn in exp.data[terr].keys():
            for i in range(exp.boot_size):
                df_list = exp.data[terr][szn]
                if exp.missing_data:
                    df_list = random.sample(
                        df_list, int((1 - exp.missing_data / 100) * len(df_list))
                    )
                try:
                    result = evaluate(
                        exp,
                        random.sample(df_list, 1)[0],
                    )
                except ValueError:
                    print(terr, szn, len(df_list))
                    break
                for stat in result.keys():
                    for model in exp.mod_names():
                        exp.results[terr][szn]["res"].loc[stat, model].ap(
                            result[stat]["res"][model].iloc[0]
                        )
                        exp.results[terr][szn]["rank"].loc[stat, model].ap(
                            result[stat]["rank"][model].iloc[0]
                        )
    print(
        f"Build complete for n={exp.boot_size}: {format(time.time() - s, '.2f')}s to run. Concatenating now..."
    )

    concatenate(exp)
    pth = f"{exp.rank_csv_path}/{date.today().strftime('%d%b')}_{exp.depth}_{exp.missing_data}.pickle"
    with open(pth, "wb") as handle:
        pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Concatenation complete.")
    print(f"Experiment stored in: {pth}")


def rank_distribution(exp, stat="", terr="", szn=""):
    print("Generating rank distribution...")
    if stat == "BIAS":
        return bias_distribution(exp, terr=terr, szn=szn)
    idx = pd.IndexSlice
    if stat == "":
        stat = [i for i in list(exp.stat_list) if i != "BIAS"]
    if terr == "":
        terr = list(set(exp.terr_list))
    if szn == "":
        szn = list(set(exp.szn_list))
    df = exp.results.loc[idx[["rank"], terr, szn, stat]].droplevel("mode")
    for mod in df.columns:
        # cell.ranks() = [1000, 0, 0, 0]gives a count for each ranking.
        # So, model that ranks 1 every time in n-1000: [1000, 0, 0, 0]
        m = [cell.ranks() for cell in list(df[mod].values)]
        df[mod] = m

    # SELECT WHAT YOU WANT TO SELECT

    lst = []
    for mod in df.columns:
        # list of four numbers that are distribution
        # num. of ranks = bootstrap size * subset length
        total_rankings = exp.boot_size * len(df)
        # rank_count = # of times model occupies a rank over all testing conditions
        rank_count = [sum(i[rank] for i in df[mod]) for rank in range(4)]
        # dist = count of rank / number of ranks
        dist = [i / total_rankings for i in rank_count]
        lst.append(dist)

    rank_dist = pd.DataFrame(
        lst, index=list(df.columns), columns=["First", "Second", "Third", "Fourth"]
    )

    return rank_dist


def bias_distribution(exp, terr="", szn=""):
    idx = pd.IndexSlice
    if terr == "":
        terr = list(set(exp.terr_list))
    if szn == "":
        szn = list(set(exp.szn_list))
    df = exp.results.loc[idx[["res"], terr, szn, ["BIAS"]]].droplevel("mode")
    total_rankings = exp.boot_size * len(df)
    total_bias = []
    for mod in df.columns:
        # cell.ranks() = [1000, 0, 0, 0]gives a count for each ranking.
        # So, model that ranks 1 every time in n-1000: [1000, 0, 0, 0]
        m = [cell.bias() for cell in list(df[mod].values)]
        df[mod] = m
        total_bias.append(sum(df[mod]) / total_rankings)

    bias_dist = pd.DataFrame(
        total_bias,
        index=list(df.columns),
        columns=["BIAS"],
    )
    # DOUBLE CHECK WHETHER RANK_DIST NEEDS TO BE .T TRANSPOSED
    return bias_dist


def concatenate(exp):
    """
    Get x2 df:
    ranks = (terr, szn, stat)
    res = (terr, szn, stat)
    """

    res, rank = [], []
    for terr in exp.results.keys():

        res_terr, rank_terr = [], []
        for szn in exp.results[terr].keys():

            res_szn = exp.results[terr][szn]["res"]
            res_szn.index.name = "stat"
            res_szn["szn"] = szn
            res_terr.append(res_szn.reset_index(drop=False))

            rank_szn = exp.results[terr][szn]["rank"]
            rank_szn.index.name = "stat"
            rank_szn["szn"] = szn
            rank_terr.append(rank_szn.reset_index(drop=False))

        res_terr = pd.concat(res_terr)
        res_terr["terr"] = terr
        res.append(res_terr)

        rank_terr = pd.concat(rank_terr)
        rank_terr["terr"] = terr
        rank.append(rank_terr)

    res = pd.concat(res)
    res["mode"] = "res"

    rank = pd.concat(rank)
    rank["mode"] = "rank"

    df = pd.concat([rank, res])
    df.set_index(["mode", "terr", "szn", "stat"], inplace=True)
    exp.results = df
