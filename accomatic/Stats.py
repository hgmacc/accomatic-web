from operator import indexOf

import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from NcReader import *
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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


def nse_one(prediction, observation):
    o_mean = observation.mean()
    a = sum(abs(observation - prediction))
    b = sum(abs(observation - o_mean))
    return 1 - (a / b)


def bias(obs, mod):
    return np.mean(mod - obs)


def rmse(obs, mod):
    return mean_squared_error(obs, mod, squared=False)


stats = {
    "RMSE": rmse,
    "R2": r2_score,
    "E1": nse_one,
    "MAE": mean_absolute_error,
    "WILL": willmott_refined_d,
    "BIAS": bias,
}

time_code_months = {
    "ALL": list(range(1, 13)),
    "DJF": [1, 2, 12],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "FREEZE": [10, 11, 12, 1, 2, 3],
    "THAW": [4, 5, 6, 7, 8, 9],
}


def bootstrap(o, m):

    pass


def run(o, m, exp, site, szn, data_avail_val):
    for sim in m.columns:
        d = {"data_avail": data_avail_val}
        for stat in exp.acco_list:
            d[stat] = stats[stat](o, m[sim])
        row = exp.res_index(site, sim, szn)
        exp.results.loc[row, list(d.keys())] = list(d.values())


def build(exp):
    for site in exp.sites_list:
        df = exp.obs(site).join(exp.mod(site)).dropna()
        o, m = df.obs.dropna(), df.drop(["obs"], axis=1)
        for szn in exp.szn_list:
            run(
                o[o.index.month.isin(time_code_months[szn])],
                m[m.index.month.isin(time_code_months[szn])],
                exp,
                site,
                szn,
                100,
            )
