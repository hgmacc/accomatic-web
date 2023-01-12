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


def read_nc(file_path) -> pd.DataFrame:
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
    odf = average_obs_site(odf)
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
    mod = read_geotop("/home/hma000/accomatic-web/tests/test_data/nc/snow_75Sites.nc")
    df = mod.join(obs)
    df["ens"] = df[["era5", "merr", "jra5"]].mean(axis=1)
    df = df.sort_index().dropna()
    return df


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
