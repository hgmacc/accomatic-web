

def heatmap_plot():
    df = pd.read_csv("/home/hma000/accomatic-web/csvs/szn_stats.csv", index_col=[0,1], names=['rmse','mae','r2'], header=0)
    all = df.index.get_level_values(1).unique()
    newdf = df.loc[(df.index.get_level_values(1) == all[0])].mean()

    for setup in all[1:]:
        newdf = pd.concat([newdf, df.loc[(df.index.get_level_values(1) == setup)].mean()], axis=1)

    newdf.columns = all
    month_to_season_dct = {
                'All-Time' : range(1, 13),
                'Winter' : [1,2,12],
                'Spring' : [3,4,5],
                'Summer' : [6,7,8],
                'Fall' : [9,10,11]
            }

    all_time = newdf.filter(regex='All-Time').T
    all_time.rmse = all_time.rmse.rank(method='max')
    all_time.mae = all_time.rmse.rank(method='max')
    all_time.r2 = all_time.rmse.rank(method='max')

    winter = newdf.filter(regex='Winter').T
    winter.rmse = winter.rmse.rank(method='max')
    winter.mae = winter.rmse.rank(method='max')
    winter.r2 = winter.rmse.rank(method='max')

    spring = newdf.filter(regex='Spring').T
    spring.rmse = spring.rmse.rank(method='max')
    spring.mae = spring.rmse.rank(method='max')
    spring.r2 = spring.rmse.rank(method='max')

    summer = newdf.filter(regex='Summer').T
    summer.rmse = summer.rmse.rank(method='max')
    summer.mae = summer.rmse.rank(method='max')
    summer.r2 = summer.rmse.rank(method='max')

    fall = newdf.filter(regex='Fall').T
    fall.rmse = fall.rmse.rank(method='max')
    fall.mae = fall.rmse.rank(method='max')
    fall.r2 = fall.rmse.rank(method='max')

    sns.color_palette(["#1CE1CE", "#008080", "#F3700E"])
    sns.set_context('poster')
    sns.despine()
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(25, 20))

    sns.heatmap(all_time, annot=True, ax=ax1, cmap="YlGnBu")
    sns.heatmap(winter, annot=True, ax=ax2, cmap="YlGnBu")
    sns.heatmap(summer, annot=True, ax=ax3, cmap="YlGnBu")
    sns.heatmap(spring, annot=True, ax=ax4, cmap="YlGnBu")
    sns.heatmap(fall, annot=True, ax=ax5, cmap="YlGnBu")
    plt.tight_layout()
    plt.savefig('plots/heatmap.png', dpi=300)


def crossplot():
    for site in odf.index.get_level_values('sitename').unique():
        # Pull out only temp data for site
        odf_site = odf.loc[(odf.index.get_level_values('sitename') == site)]
        mdf_site = mdf.loc[(mdf.index.get_level_values('sitename') == site)] 

        # Get rid of 'sitename' index
        odf_site = odf_site.reset_index(level=(1), drop=True)
        mdf_site = mdf_site.reset_index(level=(1), drop=True)

        # Clip model data to obs data
        mdf_site = clip_mod(odf_site, mdf_site)

        df = pd.concat([odf_site.soil_temperature, mdf_site.era5, mdf_site.merr, mdf_site.jra5], axis=1)
        df = df.dropna()

        df.index = pd.to_datetime(df.index)

        sns.set_context('poster')
        sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

        fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20, 20))

        sns.scatterplot(x="soil_temperature", y="era5", ax=ax1, data=df)
        sns.scatterplot(x="soil_temperature", y="merr",  ax=ax2, data=df)
        sns.scatterplot(x="soil_temperature", y="jra5",  ax=ax3, data=df)

        # Set title and labels for axes
        ax1.set(xlabel="Observations",
                ylabel="ERA5")
        ax2.set(xlabel="Observations",
                ylabel="MERRA-2")
        ax2.set(xlabel="Observations",
                ylabel="JRA-55")

        ax1.set_xlim([-40, 20])
        ax1.set_ylim([-40, 20])

        ax2.set_xlim([-40, 20])
        ax2.set_ylim([-40, 20])

        ax3.set_xlim([-40, 20])
        ax3.set_ylim([-40, 20])

        plt.savefig('plots/crossplot.png', dpi=300)
        sys.exit()


def crossplot2():
    for site in odf.index.get_level_values('sitename').unique():
        # Pull out only temp data for site
        odf_site = odf.loc[(odf.index.get_level_values('sitename') == site)]
        mdf_site = mdf.loc[(mdf.index.get_level_values('sitename') == site)] 

        # Get rid of 'sitename' index
        odf_site = odf_site.reset_index(level=(1), drop=True)
        mdf_site = mdf_site.reset_index(level=(1), drop=True)

        # Clip model data to obs data
        mdf_site = clip_mod(odf_site, mdf_site)

        df = pd.concat([odf_site.soil_temperature, mdf_site.era5, mdf_site.merr, mdf_site.jra5], axis=1)
        df = df.dropna()

        df.index = pd.to_datetime(df.index)

        sns.set_context('poster')
        sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

        fig, ax = plt.subplots(figsize=(20, 20))

        sns.scatterplot(x="soil_temperature", y="era5", data=df)
        sns.scatterplot(x="soil_temperature", y="merr", data=df)
        sns.scatterplot(x="soil_temperature", y="jra5", data=df)

        # Set title and labels for axes
        ax.set_xlim([-40, 20])
        ax.set_ylim([-40, 20])

        plt.savefig('plots/crossplot.png', dpi=300)
        sys.exit()


def plot():
    #palette = ["#1CE1CE", "#008080", "#F3700E", "#F50B00", "#59473C", "#96e3dc", "#165751"]

    pal = sns.dark_palette("#F3700E", 23)
    sns.set_context('poster')
    sns.despine()
    sns.set_style({'font.family': 'serif', 'font.serif': 'Times New Roman'})

    fig, ax = plt.subplots(figsize=(30, 13.5))

    sns.lineplot(x='time', 
                y='soil_temperature', 
                hue='sitename', 
                data=odf, 
                legend=False, 
                palette=pal)


    # Set title and labels for axes
    ax.set(xlabel="Date",
           ylabel="GST (C)")

    # Define the date format
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)

    plt.savefig("tsplot_obs.png", dpi=300, transparent=True) 



plot = True
if plot:
    df = pd.read_csv('terrains_df.csv')
    df.time = pd.to_datetime(df['time'], format="%Y-%m-%d")
    df = df[df.time.dt.month > 5]
    df = df[df.time.dt.month < 9]
    df = df.reset_index()
    df = df.set_index(['time', 'simulation'])
    # Setting custom colour palette for seaborn plots
    palette = ["#1CE1CE", "#008080", "#F3700E", "#F50B00", "#59473C"]
    sns.set_palette(sns.color_palette(palette))
    sns.set_context('poster')
    sns.despine()
    sns.set_style({'font.family':'serif', 'font.serif':'Times New Roman'})
    
    # Create figure and plot space
    fig, ax = plt.subplots(figsize=(20, 10))
    sns.lineplot(x='time', y='soil_temperature', linewidth = 4, data = df, ax=ax, hue='simulation')

    # Set title and labels for axes
    ax.set(xlabel="Date",
        ylabel="GST (C˚)")
    
    ax.legend(['Clay','Peat','Sand','Rock', 'Gravel'],loc='best')
    
    # Define the date format
    date_form = DateFormatter("%m-%Y")
    ax.xaxis.set_major_formatter(date_form)
    
    plt.savefig("terrains_thick.png", dpi=300, legend=False, transparent=True)
    

