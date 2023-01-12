import os
import re
import sys
import typing
from datetime import datetime, timedelta
from os import path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xarray as xr
from tsp import readers


def create_acco_nc(exp) -> None:
    acco = xr.Dataset(
        str(exp.model_pth + "_acco_results.nc"), mode="w", format="NETCDF4"
    )
    # build stats
    # populate file

    acco.close()


def read_geotop(file_path, sitename="") -> pd.DataFrame:
    # Get dataset
    m = xr.open_dataset(file_path, group="geotop")

    m["sitename"] = m.sitename.str.replace(r"_site", "")

    if sitename != "":
        m = m.where(m.sitename == sitename, drop=True)

    mdf = m.to_dataframe()

    # Drop dumb columns and rename things
    mdf = mdf.drop(["model", "pointid"], axis=1).rename(
        {"Date": "time", "Tg": "soil_temperature"}, axis=1
    )
    mdf = mdf.reset_index(level=("time", "soil_depth"), drop=True)
    mdf = mdf.reset_index(drop=False)

    # Fix simulation colummn
    mdf.simulation = [
        line[-12:-8] for line in mdf.simulation
    ]  # (...)led_merr_3e66cca -> merr

    # Setting up time index
    mdf.time = pd.to_datetime(mdf["time"]).dt.date

    mdf = mdf.set_index([mdf.time, mdf.sitename, mdf.simulation], append=True)
    mdf = mdf.drop(["time", "sitename", "simulation"], axis=1)
    mdf = mdf.reset_index(level=(0), drop=True)

    mdf = mdf.unstack(level=2).soil_temperature
    return mdf


def average_obs_site(odf) -> pd.DataFrame:
    """
    Averaging GST output for each site.
    """

    # KDI-E-Org_02 -> KDI-E-Org
    odf["sitename"] = odf.index.get_level_values("sitename").str[:-3]

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
