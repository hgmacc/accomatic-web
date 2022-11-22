# MUST BE RUN FROM THE ACCOMATIC MODULE IN VS-CODE (TALIK)
import sys
import os
import getopt
import re

from Experiment import *
from Stats import *
import glob



def average_obs_site(odf) -> pd.DataFrame:
    """
    Averaging GST output for each site.
    """
    
    # KDI-E-Org_02 -> KDI-E-Org 
    odf["sitename"] = odf.index.get_level_values("sitename").str.replace("\_ST\d\d$", "", regex=True)
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



ldg = read_nc('/fs/yedoma/usr-storage/hma000/obs_data/lacdegras.nc')
ldg = ldg[ldg.index.get_level_values("sitename").str.contains('NGO')]

kdi = read_nc('/fs/yedoma/usr-storage/hma000/obs_data/kdi.nc').dropna() # .reset_index(drop=False)
places = ['KDI-E-Org2', 'KDI-E-Wet', 'KDI-E-ShrubM']
kdi.drop(places, level=1, axis=0, inplace=True)

yk = read_nc('/fs/yedoma/usr-storage/hma000/obs_data/yellowknife.nc').dropna()
yk = yk[yk.index.get_level_values("sitename").str.contains('YK')]

ykl = pd.concat([yk, kdi, ldg], join="outer")

print(ykl.head())