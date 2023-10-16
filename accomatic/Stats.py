from operator import indexOf
import random
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from accomatic.NcReader import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
from static.statistics_helper import *
import time
import pickle


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

        if stat == "BIAS":
            rank = res.abs().rank(method=acco_rank[stat])
        else:
            rank = res.rank(method=acco_rank[stat], axis=1)

        stat_dict[stat]["res"] = res
        stat_dict[stat]["rank"] = rank

    return stat_dict


def build(exp):
    print("Building")
    for terr in exp.data.keys():
        for szn in exp.data[terr].keys():
            for i in range(exp.boot_size):
                s = time.time()
                df_list = exp.data[terr][szn]
                if exp.missing_data:
                    df_list = random.sample(
                        df_list, int((1 - exp.missing_data / 100) * len(df_list))
                    )

                result = evaluate(
                    exp,
                    random.sample(df_list, 1)[0],
                )

                for stat in result.keys():
                    for model in exp.mod_names():
                        exp.results[terr][szn]["res"].loc[stat, model].ap(
                            result[stat]["res"][model].iloc[0]
                        )
                        exp.results[terr][szn]["rank"].loc[stat, model].ap(
                            result[stat]["rank"][model].iloc[0]
                        )
    print(f"This took {time.time() - s}s to run.")
    print(f"Build complete for n={exp.boot_size}.")


def concatonate(exp):
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
