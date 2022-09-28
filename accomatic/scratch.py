

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
        for szn in exp.szn_list:
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
            if szn != 'ALL':
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
