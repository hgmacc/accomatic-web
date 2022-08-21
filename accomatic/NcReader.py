import sys
import typing
from os import path
import re
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset, date2num, num2date
import xarray as xr
import sklearn 


"""
VAR NAMING CONVENTIONS
*_file = file_path (str)
*_nc = <netCDF4._netCDF4.Dataset>
prefix = m(model), o(obs), a(acco)
"""


def create_acco_nc(m_file, exp):
    """
    This function taked a gtpem netcf output and adds a group for Acco results.

    :param m_file: NC file to add acco to
    :return: Nothing. File is created.
    """
    acco = Dataset(m_file, mode="w", format="NETCDF4")

    acco.createDimension("nchars", 255)

    acco.createDimension("model")  # Model-Data pairing (for ranking)
    acco.createVariable("model", chr, ("model", "nchars"))

    acco.createDimension("sitename")  # Sitename
    acco.createVariable("sitename", chr, ("sitename", "nchars"))

    acco.createDimension("acco_name")  # Accordance measure name
    acco.createVariable("acco_name", chr, ("acco_name", "nchars"))

    acco.createDimension("time_id")  # Seasons, years etc.
    acco.createVariable("time_id", int, ("time_id",))

    acco.createVariable("stats", np.float32, ("model", "sitename", "acco_name"))

    acco.comment = "Created from Accomatic"
    acco.date_created = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    acco.obs_source = f"Obs data source: {exp.obs_pth}"
    acco.mod_source = f"Model data source: {exp.model_pth}"

    acco.close()

def average_obs_site(odf):
    # KDI-E-Org_02 -> KDI-E-Org 
    odf['sitename'] = odf.index.get_level_values('sitename').str[:-3]

    # Drop sitename index so we can use new 'sitename' col to avg over non-unique sitenames
    odf = odf.reset_index(level=(1), drop=True)

    # Average 'soil_temperature' over 'sitename'
    odf['temp_site_date'] = odf.sitename + odf.index.get_level_values('time').astype(str)
    odf.soil_temperature = odf.groupby('temp_site_date')['soil_temperature'].transform('mean')

    odf = odf.drop_duplicates(subset=["temp_site_date"], keep='first') 

    # Cleaning up so df format is still (time, sitename) : soil_temperature
    odf = odf.set_index(odf.sitename, append=True)
    odf = odf.drop(['temp_site_date', 'sitename'], axis=1)
    return odf


def getdf(file):

        # Get dataset
        m = xr.open_dataset(file["mod"], group='geotop')
        mdf = m.to_dataframe()
        
        # Drop dumb columns and rename things
        mdf = mdf.drop(['model', 'pointid'], axis=1).rename({'Date': 'time', 'Tg': 'soil_temperature'}, axis=1) 
        mdf = mdf.reset_index(level=(0,1), drop=True)
        mdf = mdf.reset_index(drop=False)

        # Fix simulation and sitename colummn 
        mdf.simulation = [line[-12:-8] for line in mdf.simulation] # (...)led_merr_3e66cca -> merr
        mdf.sitename = [line[:-5] for line in mdf.sitename]  # KDI-W-Tbot_site -> KDI-W-Tbot

        # Setting up time index
        mdf.time = pd.to_datetime(mdf['time']).dt.date
        mdf = mdf.set_index([mdf.time, mdf.sitename, mdf.simulation], append=True)
        mdf = mdf.drop(['time', 'sitename', 'simulation'], axis=1)
        mdf = mdf.reset_index(level=(0), drop=True)

        # Breakup simulation data
        mdf = mdf.unstack(level=2).soil_temperature

        # Get dataset
        o = xr.open_dataset(file['obs'])
        odf = o.to_dataframe()
                
        # Clean up columns
        odf = odf.drop(['latitude', 'longitude', 'elevation', 'depth'], axis=1).rename({'platform_id': 'sitename'}, axis=1) 

        # Fix index
        odf = odf.reset_index(level=(1), drop=True)
        odf.sitename = [line.decode("utf-8") for line in odf.sitename]
        odf = odf.set_index(odf.sitename, append=True)
        odf = odf.drop(['sitename'], axis=1)
        
        # Average over sites
        odf = average_obs_site(odf)

        return(odf, mdf)

def read_manifest(manifest_file_pth):

    df = pd.read_csv(manifest_file_pth, usecols=['site', 'model', 'forcing', 'parameters'])
    df.parameters = [line.replace('/', '.').split('.')[-2] for line in df.parameters]  # '.../*/Peat.inputs --> 'Peat'
    df.forcing = [line.split('_')[1] for line in df.forcing]  # 'scaled_merra2_1h_scf1.5_kdiBoreholes' --> 'merra2'
    df['model'] = df[['model', 'forcing', 'parameters']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    df = df.drop(['forcing', 'parameters'], axis=1)

    return df




