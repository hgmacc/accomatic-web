from accomatic.nc_reader import *
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
    "R2": mean_absolute_error,
    "MAE": r2_score,
    "NSE": nse,
    "WILL": willmot_d,
    "BIAS": bias,
}

time_code_months = {
    "ALL": range(1, 13),
    "JFD": [1, 2, 12],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "FREEZE": [10, 11, 12, 1, 2, 3],
    "THAW": [4, 5, 6, 7, 8, 9],
}

def clip_mod(odf, mdf):
    """
    Clips mod_df object to obs_nc start and end time.

    :param obs_nc: netCDF4 data object
    :param mod_df: pd.Dataframe
    :return mod_df: pd.Dataframe
    """
    start, end = odf.index[0], odf.index[-1]
    mdf = mdf.loc[(mdf.index >= start) & (mdf.index <= end)]
    return mdf

def generate_stats(df, szn, acco_list):
    # Set up x and y data for analysis
    obs = df.soil_temperature
    models = df.drop(['soil_temperature'], axis=1)

    stats_dict = {}

    for mod in models:
        stats_dict[mod+szn] = map(func, acco_list)
        
        result []

        for s in acco_list:
            result.append(float(stats[s](obs, mod))
        
        stats_dict[mod+szn] = result

    return stats_dict

def build():
    stats = {}
    for site in odf.index.get_level_values("sitename").unique():
        # Pull out only temp data for site
        odf_site = odf.loc[(odf.index.get_level_values("sitename") == site)]
        mdf_site = mdf.loc[(mdf.index.get_level_values("sitename") == site)]

        # Get rid of 'sitename' index
        odf_site = odf_site.reset_index(level=(1), drop=True)
        mdf_site = mdf_site.reset_index(level=(1), drop=True)

        # Clip model data to obs data
        mdf_site = clip_mod(odf_site, mdf_site)

        # for szn in time_code_months.keys():
        for szn in time_code_months:
            # Merge into one dataframe for analysis
            df = pd.concat(
                [
                    odf_site.soil_temperature,
                    mdf_site.era5,
                    mdf_site.merr,
                    mdf_site.jra5,
                ],
                axis=1,
            )
            df = df.dropna()
            df.index = pd.to_datetime(df.index)

            df = df[df.index.month.isin(time_code_months[szn])]
            # obs = df.soil_temperature ???
            # models = df.drop(['soil_temperature'], axis=1) ???
            if site not in stats:
                stats[site] = generate_stats(df, szn)
            else:
                stats[site].update(generate_stats(df, szn))

    stats_df = pd.DataFrame.from_dict(
        {
            (site, mod): stats[site][mod]
            for site in stats.keys()
            for mod in stats[site].keys()
        },
        orient="index",
        columns=["rmse", "mae", "r2", "nse", "will", "bias"],
    )
    return stats
