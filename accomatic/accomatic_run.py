from NcReader import *
from Settings import Settings

exp = Settings("../tests/test_data/test_toml_settings.toml")
nope = False

if exp.acco and nope:
    m = Dataset(exp.model_pth)
    o = Dataset(exp.obs_pth)
    a = path.join(path.dirname(exp.model_pth), "acco.nc")

    if not path.exists(a):
        create_acco_nc(a, exp)
        print(f'New accomatic nc file created at {a}')

    simulations = read_manifest(exp.manifest)

    if exp.acco:

        print(simulations[71:74])

        print(len(simulations[72:].site.unique()))
        print(len(simulations[72:].model.unique()))

        print(len(simulations[:72].site.unique()))
        print(len(simulations[:72].model.unique()))

        #run_acco(m, o, simulations)

    #print(m['geotop']['simulation'][:3])
    #print(simulations.head())

    # 6 simulations per site ( 3 param x 2 reanalysis data types)
    # monthly average of just jra data


FILE = {'mod': '/Users/hannahmacdonell/Documents/projects/accomatic-web/tests/test_data/KDI_10CM_23Sites.nc',
        'obs': '/Users/hannahmacdonell/Documents/projects/accomatic-web/tests/test_data/KDI_10CM_23Sites.nc'}

def df_prep():
    f = xr.open_dataset(FILE['obs'])
    odf = f.soil_temperature.to_dataframe()

    # Get dataset
    f = xr.open_dataset(FILE['mod'], group='geotop')
    mdf = f.to_dataframe()
    
    # Drop dumb columns and rename things
    mdf = mdf.drop(['model', 'pointid'], axis=1).rename({'Date': 'time', 'Tg': 'soil_temperature'}, axis=1) 
    mdf = mdf.reset_index(level=(0,1), drop=True)
    mdf = mdf.reset_index(drop=False)

    # Merge simulation and sitename colummn 
    mdf.simulation = mdf.sitename  + ',' + mdf.simulation 

    # Setting up time index
    mdf.time = pd.to_datetime(mdf['time'])
    mdf.index = mdf.time
    mdf = mdf.drop(['time', 'sitename'], axis=1)

    return (mdf, odf)


def run_acco(m_nc, o_nc, mdf):
    """
    This function runs accomatic processes and populated acco nc group accordingly.
    :param m_nc: NC files with model groups and acco group
    :param o_nc: observational data for comparison
    :return: NA: Populates acco group of ncfile
    """

    rmse, r2, mae = [], [], []
    model = 'geotop'
    # 72 = 12 * (3 Model + 3 Param)
    for o_site_index in range(12):
        for m_site_index in range(3):
            m_time = m_nc[model]["Date"]

            ref = date2num(
                datetime(1970, 1, 1, 0, 0, 0), units=m_time.units, calendar="standard"
            )

            start, end = o_nc["time"][0], o_nc["time"][-1]
            m_time = m_time - ref
            time_select = np.logical_and(m_time[:] > start, m_time[:] < end)

            x = o_nc["soil_temperature"][o_site_index][:-1]
            y = m_nc[model]["Tg"][m_site_index, time_select, 0]

            rmse.append(mean_squared_error(x, y, squared=False))
            mae.append(mean_absolute_error(x, y))
            r2.append(r2_score(x, y)) 
    print(rmse, r2, mae)
    #mdf['RMSE'], mdf['R2'], mdf['MAE'] = rmse, r2, mae
    return mdf


o = xr.open_dataset('/Users/hannahmacdonell/Downloads/KDI_obs.nc')
for i in o['platform_id'][:]:
    print(i)

