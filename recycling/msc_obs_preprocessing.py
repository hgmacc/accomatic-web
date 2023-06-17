# MUST BE RUN FROM THE ACCOMATIC MODULE IN VS-CODE (TALIK)
import getopt
import glob
import os
import re
import sys

import matplotlib.pyplot as plt
import xlrd
from Experiment import *
from Stats import *
from Plotting import *

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"

palette = ["#1CE1CE", "#008080", "#F3700E", "#F50B00", "#59473C"]


def average_obs_site(odf) -> pd.DataFrame:
    """
    Averaging GST output for each site.
    """

    # KDI-E-Org_02 -> KDI-E-Org
    odf["sitename"] = odf.index.get_level_values("sitename").str.replace(
        "\_ST\d\d$", "", regex=True
    )
    odf["sitename"] = odf.sitename.str.replace("\_\d\d$", "", regex=True)

    # Drop sitename index so we can use new 'sitename' col to avg over non-unique sitenames
    odf = odf.reset_index(level=(1), drop=True)

    # Average 'soil_temperature' over 'sitename'
    odf["temp_site_date"] = odf.sitename + odf.index.get_level_values("time").astype(
        str
    )
    odf.soil_temperature = odf.groupby("temp_site_date")["soil_temperature"].transform(
        "mean"
    )

    odf = odf.drop_duplicates(subset=["temp_site_date"], keep="first")

    # Cleaning up so df format is still (time, sitename) : soil_temperature
    odf = odf.set_index(odf.sitename, append=True)
    odf = odf.drop(["temp_site_date", "sitename"], axis=1)
    odf = odf.rename(columns={"soil_temperature": "obs"})
    return odf


def read_nc(file_path, avg=True) -> pd.DataFrame:
    # Get dataset
    o = xr.open_dataset(file_path)
    odf = o.to_dataframe()

    # Clean up columns
    odf = odf.drop(["latitude", "longitude", "elevation", "depth"], axis=1).rename(
        {"platform_id": "sitename"}, axis=1
    )

    # Fix index
    odf = odf.reset_index(level=(1), drop=True)
    odf.sitename = [line.decode("utf-8") for line in odf.sitename]
    odf = odf.set_index(odf.sitename, append=True)
    odf = odf.drop(["sitename"], axis=1)

    # Average over sites
    if avg: odf = average_obs_site(odf)
    
    return odf


def get_obs_data():
    ldg = read_nc("/fs/yedoma/usr-storage/hma000/obs_data/lacdegras.nc").dropna()
    ldg = ldg[ldg.index.get_level_values("sitename").str.contains("NGO")]

    kdi = read_nc(
        "/fs/yedoma/usr-storage/hma000/obs_data/kdi.nc"
    ).dropna()  # .reset_index(drop=False)
    places = [
        "KDI-E-Org2",
        "KDI-E-Wet",
        "KDI-E-ShrubM",
    ]  # Can't remember why we're dropping these
    kdi.drop(places, level=1, axis=0, inplace=True)

    yk = read_nc("/fs/yedoma/usr-storage/hma000/obs_data/yellowknife.nc").dropna()
    yk = yk[yk.index.get_level_values("sitename").str.contains("YK")]

    df = pd.concat([yk, kdi, ldg])
    df = df.sort_index().dropna()
    return df


def get_data():
    ldg = read_nc("/fs/yedoma/usr-storage/hma000/obs_data/lacdegras.nc").dropna()
    ldg = ldg[ldg.index.get_level_values("sitename").str.contains("NGO")]

    kdi = read_nc(
        "/fs/yedoma/usr-storage/hma000/obs_data/kdi.nc"
    ).dropna()  # .reset_index(drop=False)
    places = ["KDI-E-Org2", "KDI-E-Wet", "KDI-E-ShrubM"]
    kdi.drop(places, level=1, axis=0, inplace=True)

    yk = read_nc("/fs/yedoma/usr-storage/hma000/obs_data/yellowknife.nc").dropna()
    yk = yk[yk.index.get_level_values("sitename").str.contains("YK")]

    obs = pd.concat([yk, kdi, ldg])  # .sort_index(inplace=True)
    mod = read_geotop(file_path = "/home/hma000/accomatic-web/tests/test_data/nc/snow_75Sites.nc")
    df = mod.join(obs)
    df["ens"] = df[["era5", "merr", "jra5"]].mean(axis=1)
    df = df.sort_index().dropna()
    return df


def missing_days_cluster():
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10,10), squeeze=True)
    fig.suptitle('Missing data plot for each NWT supersite cluster.')

    df = read_nc('/home/hma000/accomatic-web/tests/test_data/nc/ykl_obs.nc', avg=False)
    df = df[df.index.get_level_values("sitename").str.contains("AIR")==False]
    df = df[df.index.get_level_values("sitename").str.contains("_")==True]
    # Drop rows that don't contain an '_'

    #df = df.reset_index(level=['sitename'])
    df = df.rename_axis(index=('time', None))
    df = df.soil_temperature.unstack(level=1)
    times = pd.date_range(start=pd.to_datetime("2015-07-04"), end=pd.to_datetime("2021-08-18"), freq="1D").date
    df = df.reindex(times)
    
    # YK
    plt.subplot(int('311'))
    yk_col = [col for col in df.columns if 'YK' in col]
    a = ((len(df[yk_col].columns) - df[yk_col].isnull().sum(axis=1)) / len(df[yk_col].columns) * 100 )
    plt.plot(a, c="#008080")
    plt.title("Yellowknife")   
        
    # KDI
    plt.subplot(312)
    kdi_col = [col for col in df.columns if 'KDI' in col]
    a = ((len(df[kdi_col].columns) - df[kdi_col].isnull().sum(axis=1)) / len(df[kdi_col].columns) * 100 )
    plt.plot(a, c="#F50B00")
    plt.ylabel("Percentage of Available Data")
    plt.title("KDI")    
    
    # LDG
    plt.subplot(313)
    ldg_col = [col for col in df.columns if 'NGO' in col]
    a = ((len(df[ldg_col].columns) - df[ldg_col].isnull().sum(axis=1)) / len(df[ldg_col].columns) * 100 )
    plt.plot(a, c="#F3700E")
    plt.title("Lac De Gras")    
    
    plt.savefig('/home/hma000/accomatic-web/accomatic/prototype_bootstrap/plots/missing_data/missing_barplot_reindexed.png')



def missing_days_data():
    df = read_nc('/home/hma000/accomatic-web/tests/test_data/nc/ykl_gst_obs.nc', avg=False)
    df = df[df.index.get_level_values("sitename").str.contains("AIR")==False]
    df = df[df.index.get_level_values("sitename").str.contains("_")==True]
    # Drop rows that don't contain an '_'

    #df = df.reset_index(level=['sitename'])
    df = df.rename_axis(index=('time', None))
    df = df.soil_temperature.unstack(level=1)
    times = pd.date_range(start=pd.to_datetime("2015-07-04"), end=pd.to_datetime("2021-08-18"), freq="1D").date
    df = df.reindex(times)

    yk_col = [col for col in df.columns if 'YK' in col]
    kdi_col = [col for col in df.columns if 'KDI' in col]
    ldg_col = [col for col in df.columns if 'NGO' in col]
    
    dic = {}
    for cluster in [yk_col, kdi_col, ldg_col]:
        b = df[cluster]
        for col in b.columns:
            tmp = b[col].loc[b[col].first_valid_index():b[col].last_valid_index()]
            dic[col] = tmp.isnull().sum() / len(tmp) * 100

    tmp = pd.DataFrame.from_dict(dic, orient='index', columns=['percent_miss'])
    print(len(tmp[tmp.percent_miss > 0]))
    print(tmp[tmp.percent_miss > 0].mean())






def missing_days_terr(exp):
    

    num_plots = len(set(exp.terr_list))

    fig, axs = plt.subplots(num_plots, 1, sharex=True, figsize=(10, num_plots+10))#, squeeze=True)
    fig.suptitle('Missing data plot for each terrain type.')

    df = read_nc('/home/hma000/accomatic-web/tests/test_data/nc/ykl_obs.nc', avg=True)
    
    # Pull out only dat
    terr_list = []
    for i in df.index.get_level_values(1):
        try: terr_list.append(exp.terr_dict()[i])
        except KeyError:
            terr_list.append(-1)
    
    df['terrain'] = terr_list
    palette = get_color_gradient("#b3e0dc", "#036c5f", num_plots)
    for i in range(1,num_plots+1):
        a = df[df.terrain == str(i)].drop(["terrain"], axis=1)
        a = a.rename_axis(index=('time', None))
        a = a.obs.unstack(level=1)
        times = pd.date_range(start=pd.to_datetime("2015-07-04"), end=pd.to_datetime("2021-08-18"), freq="1D").date
        a = a.reindex(times)
        
        plt.title(f"Terrain No. {i}")   
        plt.subplot(int(f'{num_plots}1{i}'))
        a = ((len(a.columns) - a.isnull().sum(axis=1)) / len(a.columns) * 100 )
        plt.plot(a, c=palette[int(i)-1], label=f"Terrain No. {i}")
    plt.savefig('/home/hma000/accomatic-web/accomatic/prototype_bootstrap/plots/missing_data/missing_terr_plot_notsqueezed.png')


e = Experiment('/home/hma000/accomatic-web/tests/test_data/toml/MAR_NWT.toml')
missing_days_data()

plot = False
if plot:
    l = pd.read_excel(
        "/home/hma000/storage/terrain_exp/terrain_types.xlsx",
        sheet_name="YKL",
        usecols=["sitename", "class"],
    )
    m = pd.read_excel(
        "/home/hma000/storage/terrain_exp/terrain_types.xlsx",
        sheet_name="dict",
        usecols=["class", "title"],
    )

    classes = {
        1: l[l["class"] == 1].sitename.tolist(),
        2: l[l["class"] == 2].sitename.tolist(),
        3: l[l["class"] == 3].sitename.tolist(),
        4: l[l["class"] == 4].sitename.tolist(),
    }

    # This merges excel info to auto count the number of sites in each class and add to legend / title info.
    titles = dict(
        zip(
            m["class"],
            [
                x + " (n = %s)" % len(classes[c])
                for x, c in zip(m.title.tolist(), classes.keys())
            ],
        )
    )

    fig, ax = plt.subplots(figsize=(15, 8))
    df = get_data()
    for c in classes.keys():
        a = (
            df.loc[df.index.get_level_values("sitename").isin(classes[c])]
            .obs.dropna()
            .unstack(level=1)
        )
        a["min"] = a.min(axis=1)
        a["max"] = a.max(axis=1)
        a["mean"] = a.mean(axis=1)
        a.index = pd.to_datetime(a.index)
        a = a[["min", "mean", "max"]].resample("W").mean()
        a = a[a.index.year == 2020]
        plt.fill_between(
            a.index,
            np.array(a["min"], dtype=float),
            np.array(a["max"], dtype=float),
            color=palette[c],
            alpha=0.70,
        )
        plt.plot(a.index, a["mean"], color=palette[c], label=titles[c])
        plt.legend()
        plt.title("Weekly average GST values in NWT Tundra (n = 75)")
        if c == 4:
            plt.savefig("/home/hma000/storage/terrain_exp/plot_%s.png" % c)
        # plt.clf()
