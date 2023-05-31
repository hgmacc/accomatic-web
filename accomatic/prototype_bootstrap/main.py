from boot import *

def setup(sim):
    EXP = Experiment("/home/hma000/accomatic-web/tests/test_data/toml/MAR_NWT.toml")
    
    site = 'NGO-DD-2032'
    if site:
        DF = EXP.mod('NGO-DD-2032').join(EXP.obs('NGO-DD-2032')).dropna().reset_index()
        DF.index = pd.to_datetime(DF.time)
    if not site:
        DF = EXP.mod().join(EXP.obs()).dropna().groupby('time').mean()
        DF = DF.reset_index()
        DF.index = pd.to_datetime(DF.time)
        
    res = [x_day_boot(DF[DF.index.month == i], sim, 'MAE') for i in range(1, 13)]
    boot_vioplot(np.array(res), 'NGO-DD-2032', 'MAE', sim, [str(i) for i in range(1, 13)], title=f'bs_MAE_{sim}_seasonal')

if __name__ == "__main__":    
    for i in ['era5','jra55','merra2','ens']:
        setup(i)
    sys.exit()
    # Settings
    stat = 'MAE'
    site = "NGO-DD-2032"
    sim = 'ens'
    rep_list = [i for i in range(0, 201, 50)]

    # Getting data
    df = EXP.mod(site).join(EXP.obs(site)).dropna()
    
    # Setting up xticks 
    a = len(df.ens)
    xtix = [f'[{int(i/a*100)}%]' for i in rep_list]
    
    sys.exit()
    # crazy boot
    data = [crazy10_day_boot(df=df, chunk_size=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, xtix, title='bs_r_MAE_crazy_10_day')

    # bootstrapped (1) 10-day interval
    data = [simple_10_day_boot(df=df, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, xtix, title='bs_r_MAE_10_day')
    
    # entire timeseries is bootstrapped
    data = [simple_boot(df=df, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, xtix, title='bs_r_MAE_entire_ts')
    
