from NcReader import *
import xarray as xr
import seaborn as sns
from matplotlib.dates import DateFormatter


remote_file = {'mod': '/home/hma000/accomatic-web/tests/test_data/KDI_10CM_23Sites.nc',
        'obs': '/fs/yedoma/usr-storage/hma000/KDI/KDI_obs.nc'}
local_file = {'mod': '/Users/hannahmacdonell/Documents/projects/accomatic-web/tests/test_data/KDI_10CM_23Sites.nc',
        'obs': '/Users/hannahmacdonell/Documents/projects/accomatic-web/tests/test_data/KDI_obs.nc'}
(odf, mdf) = getdf(local_file)

print("-------------------------------------------------------------------")
print('\n\n')


s, e = odf.index[0], odf.index[-1]
#print(s[0], e[0])

#test = odf.query("sitename == 'KDI-E-Org2_01'")

def plot():
    palette = ["#1CE1CE", "#008080", "#F3700E", "#F50B00", "#59473C"]
    sns.set_palette(sns.color_palette(palette))
    sns.set_context('poster')
    sns.despine()

    fig, ax = plt.subplots(figsize=(30, 12))

    sns.lineplot(x='time', y='soil_temperature', hue='sitename', data=odf, legend=False)

    # Set title and labels for axes
    ax.set(xlabel="Date",
        ylabel="GST (C)")

    # Define the date format
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rc('font', size=24)
    plt.savefig("obs2.png", dpi=300, transparent=True) 

plot()

def create_acco_df(mod_nc):
    """
    Generate model stat summary df from simulations nc file.
    
    :param mod_nc: a netcdf4 data object
    :return acco_df: pd.Dataframe
    """
    simulation = mod_nc.simulation.to_dataframe().simulation.to_list()
    sim_id = [line.split('_')[-1] for line in simulation]
    forcing = [line.split('_')[3] for line in simulation]
    site = [line.split('_site')[0] for line in mod_nc.sitename.to_dataframe().sitename.to_list()]
    acco_df = pd.DataFrame(columns=['MAE', 'RMSE', 'R2'], index=[sim_id, forcing, site])
    return acco_df

def get_time_window(obs_nc):
    """
    Helper function for clip_mod().
    Returns start and end time of observational dataset. 
    
    :param obs_nc: a netcdf4 data object
    :return tuple(Datetime): 
    """
    return obs_nc.time[0], obs_nc.time[-1]


def clip_mod(obs_nc, mod_df):
    """
    Clips mod_df object to obs_nc start and end time. 
        
    :param obs_nc: netCDF4 data object
    :param mod_df: pd.Dataframe 
    :return mod_df: pd.Dataframe 
    """
    start, end = get_time_window(obs_nc)

    mod_df.time = mod_df.time - date2num(datetime(1970, 1, 1, 0, 0, 0), units='days since 1-1-1 0:0:0', calendar="standard")
    
    # Line below is wrong --> rewite for Dataframe when you have WIFI
    # time_select = np.logical_and(mod_df.time[:] > start, mod_df.time[:] < end)
    
    return mod_df

def generate_stats(x, y):
    rmse = mean_squared_error(x, y, squared=False)
    mae = mean_absolute_error(x, y)
    r2 = r2_score(x, y)
    
    stats_dict = {'RMSE' : rmse, 'mae': mae, 'r2': r2}
    
    return stats_dict


"""
N sites
obs = N sites
mod = M mods

sims = M mods x N sites

23 sites
obs = 23 sites
mod = 3 mods x 23 sites 

simulations = 3 mods x 23 sites = 69 sims

for site in obs:
    find start & end time (function)
    get M sims 
    clip to start & end time
    add stat results to df columns 
    
DF

                                    RMSE       MAE      R2
site            model     szn
KDI-W-Org1      merr     winter     xx          xx      xx        
KDI-W-Org2      era5     summer     xx          xx      xx
KDI-W-Org3      jra      spring     xx          xx      xx

Multi-index dataframe: season, model, terrain type ("site")
"""
