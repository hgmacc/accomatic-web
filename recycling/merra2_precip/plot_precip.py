import pandas as pd
import glob
from read_data import *
import sys
sites_pth = '/home/hma000/storage/merra_precip_test/ykl_csvs/sites.csv'
sites = get_sites(sites_pth)
site_dict = dict(zip(sites.station_number, sites.station_name))

data1 = {'Downloaded': glob.glob('/home/hma000/storage/merra_precip_test/ykl_csvs/*downloaded.csv'),
        'Interpolated': glob.glob('/home/hma000/storage/merra_precip_test/ykl_csvs/*interpolated.csv'), 
        'Scaled': glob.glob('/home/hma000/storage/merra_precip_test/ykl_csvs/*scaled.csv')}
merra_data = {'Downloaded': '/home/hma000/storage/merra_precip_test/ykl_csvs/merra2_downloaded.csv',
        'Interpolated': '/home/hma000/storage/merra_precip_test/ykl_csvs/merra2_interpolated.csv', 
        'Scaled': '/home/hma000/storage/merra_precip_test/ykl_csvs/merra2_scaled.csv'}
kdi_test = {'Downloaded': glob.glob('/home/hma000/storage/merra_precip_test/KDI_test/csvs/*downloaded.csv'),
        'Interpolated': glob.glob('/home/hma000/storage/merra_precip_test/KDI_test/csvs/*interpolated.csv'), 
        'Scaled': glob.glob('/home/hma000/storage/merra_precip_test/KDI_test/csvs/*scaled.csv')}

def get_colour(f):
    if 'merra2' in os.path.basename(f):
        return "#F3700E"
    elif 'jra55' in os.path.basename(f):
        return  "#008080"
    elif 'era5' in os.path.basename(f):
        return "#F50B00"
    else:
        return (f'{f} is not a pth with a colour.')

def all_precip_plots(sites, data):
    """
    Plots precipitation for five sites throughout YKL clusters for: downloaded, interpolated and scaled reanalysis data. 
    JRA55 MERRA2 and ERA5 all plotted. 
    """
    for site in sites.station_name:
        # Create figure with three plots: downloaded/interpolated/scaled
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharey=True, figsize=(10,10), squeeze=True)
        fig.tight_layout(pad=4.0)
        fig.suptitle(f'Precipitation at {site}', fontsize=16)

        a = list(data.keys())
        print('/n/n Site:')
        # Downloaded
        for f in data[a[0]]:  
            df = pd.read_csv(f, parse_dates=True, index_col='time')
            df = df[df.index.year == 2018]
            ax0.plot(df[site], label=os.path.basename(f).split('_')[0], c=get_colour(f))
            ax0.legend()
            ax0.set_title(a[0])
            if os.path.basename(f) == 'merra2_downloaded.csv':
                print(df.head(1))
        
        # Interpolated
        for f in data[a[1]]:
            df = pd.read_csv(f, parse_dates=True, index_col='time')
            if os.path.basename(f) == 'jra55_interpolated.csv':
                df[site] = df[site] / 86400
            print(df.head())
            df = df[df.index.year == 2018]
            ax1.plot(df[site], c=get_colour(f))
            ax1.set_ylabel('Weekly Precip Average (kg m-2 s-1)')
            ax1.set_title(a[1])
        
        for f in data[a[2]]:
            df = pd.read_csv(f, parse_dates=True, index_col='time')
            #df = df[pd.to_datetime(df.index).year == 2019]
            df = df[df.index.year == 2018]
            ax2.plot(df[site], c=get_colour(f))
            ax2.set_title(a[2])
            if os.path.basename(f) == 'merra2_scaled.csv':
                print(df.head(1))
                
        plt.ylim([0, 0.00035])
        plt.xlabel('Time')
        plt.xticks(rotation=70)
        plt.savefig(f'/home/hma000/accomatic-web/accomatic/merra2_precip/plots/13MAR/{site}.png')
        plt.clf()
        plt.close()

def globsim_scf():   
    """
    Plots a bunch of different precip data given varied snow correction factors stipulated in the GlobSim toml file.
    """
    paths = {'MERRA': glob.glob('/home/hma000/storage/SCF_test/SCF*/mer*'),
             'JRA55': glob.glob('/home/hma000/storage/SCF_test/SCF*/jra*'),
             'ERA5': glob.glob('/home/hma000/storage/SCF_test/SCF*/era*')}
    paths = glob.glob('/home/hma000/storage/SCF_test/SCF*/*.nc')
    sites = get_sites(sites_pth='/home/hma000/storage/SCF_test/par/ykl.csv', opt='dic')


    for site in sites.values():
        plt.figure(figsize = (20, 6), dpi = 80)

        for f in paths:
            df = get_nc_point_data(f, sites)
            if f.split('/')[-2] == 'SCF_0.5':
                plt.plot(df[site], label='_none_', c=get_colour(f), alpha=0.30)
            elif f.split('/')[-2] == 'SCF_1.0':
                plt.plot(df[site], label='_none_', c=get_colour(f), alpha=0.60)
            elif f.split('/')[-2] == 'SCF_1.5':
                plt.plot(df[site], label=os.path.basename(f).split('_')[1], c=get_colour(f))
            else:
                sys.exit('No SCF findable.')

        plt.legend()
        plt.xlabel('Time')
        plt.xticks(rotation=70)
        plt.title(f'Precipitation at {site}', fontsize=12)

        plt.savefig(f'/home/hma000/storage/SCF_test/plots/{site}.png')
        plt.clf()
        plt.close()
        print(site)
# Plot all merra data for all five sites
def all_merra_precip():
    """
    Produces one figure with three subplots (downloaded, interpolated, and scaled globsim files) with precip data for five different sites.
    """
    # Create figure with three plots: downloaded/interpolated/scaled
    fig, (ax0, ax1, ax2) = plt.subplots(3, 1, sharey=True, figsize=(10,10), squeeze=True)
    fig.tight_layout(pad=4.0)
    fig.suptitle(f'MERRA-2 Precipitation', fontsize=16)
    
    # Downloaded        
    df = pd.read_csv(merra_data['Downloaded'], parse_dates=True, index_col='time')
    df = df[df.index.year == 2018]
    for site in sites.station_name:
        ax0.plot(df[site], label=site)
    ax0.legend()
    ax0.set_title('Downloaded')
        
    # Interpolated        
    df = pd.read_csv(merra_data['Interpolated'], parse_dates=True, index_col='time')
    df = df[df.index.year == 2018]
    for site in sites.station_name:
        ax1.plot(df[site], label=site)
    ax1.set_title('Interpolated')
        
    # Scaled        
    df = pd.read_csv(merra_data['Scaled'], parse_dates=True, index_col='time')
    df = df[df.index.year == 2018]
    for site in sites.station_name:
        ax2.plot(df[site], label=site)
    ax2.set_title('Scaled')

    plt.ylim([0, 0.00035])
    plt.xlabel('Time')
    plt.xticks(rotation=70)
    plt.savefig(f'/home/hma000/accomatic-web/accomatic/merra2_precip/plots/20FEB/merra2_precip.png')

def swap_merra_precip_plot():
    """
    Plots precipitation data from multiple differend scaled globsim files/ 
    """
    new_mer = xr.open_mfdataset('/home/hma000/storage/yk_kdi_ldg/scaled/ykl/scaled_merra2_1h_scf1.0.nc', engine="netcdf4")
    era = xr.open_mfdataset('/home/hma000/storage/yk_kdi_ldg/scaled/ykl/scaled_jra55_1h_scf1.0.nc', engine="netcdf4")
    jra = xr.open_mfdataset('/home/hma000/storage/yk_kdi_ldg/scaled/ykl/scaled_era5_1h_scf1.5.nc', engine="netcdf4")

    for i in range(14):
        # Daily Avg
        # old_mer.PREC_sur.isel(station=i).groupby('time.day').mean().plot.line(color="purple",  label='old_mer')

        jra.PREC_sur.isel(station=i).plot.line(color="black",  label='jra')
        new_mer.PREC_sur.isel(station=i).plot.line(color="purple",  label='new_mer')
        era.PREC_sur.isel(station=i).plot.line(color="blue",  label='era')

        plt.legend()
        plt.savefig(f'accomatic/merra2_precip/plots/21FEB/swap_merra_precip_plot_stn{i+1}.png')
        plt.clf()
        plt.close()
        
    new_mer.close()
    era.close()
    jra.close()
    new_mer.close()
    
def temp_merra_precip_plot():
    """
    Plots precipitation data from multiple differend scaled globsim files/ 
    """

    mer_1 = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/ykl_ncs/ts_merra2_scaled.nc', engine="netcdf4")
    mer_2 = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/yk_merra_bandaid/MERRA2_fixed.nc', engine="netcdf4")
    mer_3 = xr.open_mfdataset('/home/hma000/storage/merra_precip_test/temp/scaled_merra2_1h_scf3.0.nc', engine="netcdf4")
    for i in range(14):
        # Daily Avg -> .groupby('time.day').mean()
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.plot(mer_1.AIRT_pl.isel(station=i).groupby('time.day').mean(), color="purple",  label='mer_1', alpha=0.5)
        ax.plot(mer_2.AIRT_pl.isel(station=i).groupby('time.day').mean(), color="blue",  label='mer_2', alpha=0.5)
        ax.plot(mer_3.AIRT_pl.isel(station=i).groupby('time.day').mean(), color="red",  label='mer_3', alpha=0.5, linestyle='dotted')
        ax.set_ylabel('temperature')
        
        ax2=ax.twinx()
        ax2.plot(mer_1.PREC_sur.isel(station=i).groupby('time.day').mean(), color="purple",  label='mer_1', alpha=0.5)
        ax2.plot(mer_2.PREC_sur.isel(station=i).groupby('time.day').mean(), color="blue",  label='mer_2', alpha=0.5)
        ax2.plot(mer_3.PREC_sur.isel(station=i).groupby('time.day').mean(), color="red",  label='mer_3', alpha=0.5, linestyle='dotted')

        ax2.set_ylabel("precip")
        
        plt.legend()
        plt.savefig(f'accomatic/merra2_precip/plots/06MAR/temp/merra_{i+1}.png')
        plt.clf()
        plt.close()
        
    mer_1.close()
    mer_2.close()
    mer_3.close()

