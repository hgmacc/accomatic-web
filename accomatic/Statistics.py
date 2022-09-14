from NcReader import *
import xarray as xr
import seaborn as sns
from matplotlib.dates import DateFormatter
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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


def generate_stats(df, szn):
    # Set up x and y data for analysis
    obs = df.soil_temperature
    models = df.drop(['soil_temperature'], axis=1)

    stats_dict = {}

    # For era5, merr, jra5:
    for mod in models:
        rmse = mean_squared_error(obs, models[mod], squared=False)
        mae = mean_absolute_error(obs, models[mod])
        r2 = r2_score(obs, models[mod])
        nse = (1-(np.sum((models[mod]-obs)**2)/np.sum((obs-np.mean(obs))**2)))
        will = np.mean((1-(((obs-models[mod])**2)/(abs(models[mod]-np.mean(obs))+abs(obs-np.mean(obs))**2)))) # this could be super wrong
        bias = (np.mean(models[mod]-obs))
        stats_dict[mod+'-'+szn] = [rmse, mae, r2, nse, will, bias]
    return stats_dict



def szn_build():
    stats = {}
    for site in odf.index.get_level_values('sitename').unique():
        # Pull out only temp data for site
        odf_site = odf.loc[(odf.index.get_level_values('sitename') == site)]
        mdf_site = mdf.loc[(mdf.index.get_level_values('sitename') == site)] 
 
        # Get rid of 'sitename' index
        odf_site = odf_site.reset_index(level=(1), drop=True)
        mdf_site = mdf_site.reset_index(level=(1), drop=True)

        # Clip model data to obs data
        mdf_site = clip_mod(odf_site, mdf_site)
        month_to_season_dct = {
                'All-Time' : range(1, 13),
                'Winter' : [1,2,12],
                'Spring' : [3,4,5],
                'Summer' : [6,7,8],
                'Fall' : [9,10,11]
            }

        #for szn in month_to_season_dct.keys():
        for szn in month_to_season_dct:
            # Merge into one dataframe for analysis
            df = pd.concat([odf_site.soil_temperature, mdf_site.era5, mdf_site.merr, mdf_site.jra5], axis=1)
            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            
            df = df[df.index.month.isin(month_to_season_dct[szn])]

            if site not in stats: stats[site] = generate_stats(df, szn)
            else: stats[site].update(generate_stats(df, szn))


    stats_df = pd.DataFrame.from_dict({(site,mod): stats[site][mod] 
                                        for site in stats.keys() 
                                        for mod in stats[site].keys()},
                                        orient='index', 
                                        columns=['rmse', 'mae', 'r2', 'nse', 'will', 'bias'])
    stats_df.to_csv("/home/hma000/accomatic-web/csvs/13SEP_KDI.csv")


"""    

newdf = pd.concat([df.loc[(df.index.get_level_values(1) == 'era5')].mean(), 
                   df.loc[(df.index.get_level_values(1) == 'merr')].mean(),
                   df.loc[(df.index.get_level_values(1) == 'jra5')].mean()], axis=1)
newdf = newdf.rename({0: 'era5', 1: 'merr', 2: 'jra5'}, axis=1)


DF

                                    RMSE       MAE      R2
site            model     szn
KDI-W-Org1      merr     winter     xx          xx      xx        
KDI-W-Org2      era5     summer     xx          xx      xx
KDI-W-Org3      jra      spring     xx          xx      xx

Multi-index dataframe: season, model, terrain type ("site")
"""
