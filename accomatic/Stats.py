from operator import indexOf
from NcReader import *
import xarray as xr
import seaborn as sns
from matplotlib.dates import DateFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def nse(obs, mod):
    return 1 - (np.sum((mod - obs) ** 2) / np.sum((obs - np.mean(obs)) ** 2))


# this could be super wrong
def willmot_d(obs, mod):
    return np.mean(
        (
            1
            - (
                ((obs - models[mod]) ** 2)
                / (abs(models[mod] - np.mean(obs)) + abs(obs - np.mean(obs)) ** 2)
            )
        )
    )


def bias(obs, mod):
    return np.mean(mod - obs)


def rmse(obs, mod):
    return mean_squared_error(obs, mod, squared=False)


stats = {
    "RMSE": rmse,
    "R2": r2_score,
    "MAE": mean_absolute_error,
    "NSE": nse,
    "WILL": willmot_d,
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
