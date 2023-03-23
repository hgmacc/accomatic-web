from boot import *


if __name__ == "__main__":    
    # Settings
    stat = 'MAE'
    site = "NGO-DD-2035"
    sim = 'ens'
    rep_list = [i for i in range(0, 251, 50)]

    # Getting data
    df = EXP.mod(site).join(EXP.obs(site)).dropna()
    df['ens'] = df[['era5','merra2','jra55']].mean(axis=1)
    
    df = remove_days(df, chunk_size=250, reps=1)
    plot_ts_missing_days(df, site)

    sys.exit()
    
    
    # Setting up xticks 
    a = len(df.ens)
    xtix = [f'[{int(i/a*100)}%]' for i in rep_list]
 
    # bootstrapped (1) 10-day interval
    data = [simple_10_day_boot(df=df, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, xtix, title='bs_r_MAE_10_day')
    
    # entire timeseries is bootstrapped
    data = [simple_boot(df=df, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, xtix, title='bs_r_MAE_entire_ts')
    
