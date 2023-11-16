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
    arr = np.array([i.v for i in df_col.to_list()])
    return Data(np.nanmean(arr, axis=0))


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


def willmott_refined_d(obs, mod):
    # From Luis (2020)
    o_mean = obs.mean()
    a = sum(abs(mod - obs))
    b = 2 * sum(abs(mod - o_mean))
    if a <= b:
        return 1 - (a / b)
    else:
        return (b / a) - 1


def r_score(obs, mod):
    return np.corrcoef(obs, mod)[0][1]


def nse_one(prediction, observation):
    o_mean = observation.mean()
    a = sum(abs(observation - prediction))
    b = sum(abs(observation - o_mean))
    return 1 - (a / b)


def bias(obs, mod):
    return np.mean(mod - obs)


def rmse(obs, mod):
    return mean_squared_error(obs, mod, squared=False)


acco_measures = {
    "RMSE": rmse,
    "R": r_score,
    "E1": nse_one,
    "MAE": mean_absolute_error,
    "WILL": willmott_refined_d,
    "BIAS": bias,
}

acco_rank = {
    "RMSE": "min",
    "R": "max",
    "E1": "max",
    "MAE": "min",
    "WILL": "max",
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


def get_block(df_list):
    """
    Takes list of terrain-szn-plot dataframes.
    Returns timeseries of length n where:
        n = len_of_obs / b
        b = window (10 days)
    """
    b = 10
    n = int(round(sum([len(df_i) for df_i in df_list]) / b))

    block_ts = []
    for i in range(n):
        df = df_list[random.randint(0, len(df_list) - 1)].reset_index(drop=False)
        nrows = range(df.shape[0])
        ix = random.randint(nrows.start, nrows.stop - (b + 1))
        block_ts.append(df.iloc[ix : ix + b].set_index("Date"))
    block_ts = pd.concat(block_ts)
    return block_ts


def evaluate(exp, block):

    stat_dict = {
        stat: {"res": pd.DataFrame, "rank": pd.DataFrame} for stat in exp.acco_list
    }
    for stat in exp.acco_list:
        res = {mod: [] for mod in exp.mod_names()}
        # {'era5': 0.7, 'jra55': 0.98,  'merra': 0.25,  'ens': 0.10}
        # {'era5': 3,   'jra55': 4,     'merra': 2,     'ens': 0.4}
        for model in exp.mod_names():
            res[model].append(acco_measures[stat](block.obs, block[model]))
        res = pd.DataFrame.from_dict(res)

        if stat == "BIAS" or stat == "MAE":
            rank = res.abs().rank(method="min", axis=1)
        if stat == "WILL" or stat == "R":
            rank = res.rank(method="max", axis=1)

        stat_dict[stat]["res"] = res
        stat_dict[stat]["rank"] = rank

    return stat_dict


# {'MAE': {'res':        era5      jra55     merra2        ens
# 0  2.673795  26.327093  18.709467  14.214704, 'rank':    era5  jra55  merra2  ens
# 0   1.0    4.0     3.0  2.0}, 'BIAS': {'res':        era5      jra55     merra2        ens
# 0 -2.392443  26.327093  18.709467  14.214704, 'rank':    era5  jra55  merra2  ens
# 0   1.0    1.0     1.0  1.0}, 'WILL': {'res':        era5  jra55  merra2  ens
# 0  0.609567    0.5     0.5  0.5, 'rank':    era5  jra55  merra2  ens
# 0   4.0    1.0     2.0  3.0}}


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
                    sys.exit()
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
    print("Concatenation complete. Generating rank distribution...")
    rank_distribution(exp)
    bias_distribution(exp)

    idx = pd.IndexSlice
    exp.results.loc[idx[["res"], :, :, :]].to_pickle(
        f"{exp.rank_csv_path}/{date.today()}_results.pickle"
    )


def rank_distribution(exp):
    idx = pd.IndexSlice
    df = exp.results.loc[idx[["rank"], :, :, ["MAE"]]].droplevel("mode")
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

    exp.rank_dist = rank_dist


def bias_distribution(exp):
    idx = pd.IndexSlice
    df = exp.results.loc[idx[["res"], :, :, ["BIAS"]]].droplevel("mode")
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
    exp.bias_dist = bias_dist


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
