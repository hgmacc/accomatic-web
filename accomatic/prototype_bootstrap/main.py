from boot import *


if __name__ == "__main__":    
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
    
    # crazy boot
    data = [crazy10_day_boot(df=df, chunk_size=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, xtix, title='bs_r_MAE_crazy_10_day')
    sys.exit()
    # bootstrapped (1) 10-day interval
    data = [simple_10_day_boot(df=df, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, xtix, title='bs_r_MAE_10_day')
    
    # entire timeseries is bootstrapped
    data = [simple_boot(df=df, reps=i) for i in rep_list]
    boot_vioplot(data, site, stat, sim, xtix, title='bs_r_MAE_entire_ts')
    


