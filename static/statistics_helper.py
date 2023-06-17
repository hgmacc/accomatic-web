from operator import indexOf
import random
import seaborn as sns
import xarray as xr
from matplotlib.dates import DateFormatter
from accomatic.NcReader import *
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
        return(spl(np.linspace(-1, 1, 300)))
    
    def __repr__(self):
        return repr(list(self.v))



def average_data(df_col):
    # df_col: column of np.arrays
    arr = np.array([i.v for i in df_col.to_list()])    
    return Data(np.nanmean(arr, axis=0))


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
    "DEC": [12]
}
