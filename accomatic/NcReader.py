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


def plot(m_nc, o_nc, site_index: int):
    """
    This function plots simulated vs observed temperatures.
    Plot is clipped to obs time extent.

    :param m_nc: netCDF4._netCDF4.Dataset of simulated temperature
    :param o_nc: netCDF4._netCDF4.Dataset of observations
    :param site_index: index of site you'd like to plot
    """

    # Get temp
    m_temp = m_nc["Tg"]
    o_temp = o_nc["soil_temperature"]

    # Get time
    m_time = m_nc["Date"]
    o_time = o_nc["time"]

    ref = date2num(
        datetime(1970, 1, 1, 0, 0, 0), units=m_time.units, calendar="standard"
    )

    start, end = o_time[0], o_time[-1]
    m_time = m_time - ref
    time_select = np.logical_and(m_time[:] > start, m_time[:] < end)

    data = m_temp[site_index, time_select, 1]

    plt.plot(mdates.num2date(m_time[time_select]), data[:], label='Model')
    plt.plot(mdates.num2date(o_time[:]), o_temp[site_index], label='Observations')
    plt.legend()
    plt.show()


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


def run_acco(m_nc, o_nc, df):
    """
    This function runs accomatic processes and populated acco nc group accordingly.
    :param m_nc: NC files with model groups and acco group
    :param o_nc: observational data for comparison
    :return: NA: Populates acco group of ncfile
    """

    rmse, r2, mae = [], [], []
    model = 'geotop'
    # 72 = 12 * (3 Model + 3 Param)
    for o_site_index in range(12):
        for m_site_index in range(3):
            m_time = m_nc[model]["Date"]

            ref = date2num(
                datetime(1970, 1, 1, 0, 0, 0), units=m_time.units, calendar="standard"
            )

            start, end = o_nc["time"][0], o_nc["time"][-1]
            m_time = m_time - ref
            time_select = np.logical_and(m_time[:] > start, m_time[:] < end)

            x = o_nc["soil_temperature"][o_site_index][:-1]
            y = m_nc[model]["Tg"][m_site_index, time_select, 0]

            rmse.append(metrics.mean_squared_error(x, y, squared=False))
            mae.append(metrics.mean_absolute_error(x, y))
            r2.append(metrics.r2_score(x, y))

    df['RMSE'], df['R2'], df['MAE'] = rmse, r2, mae

    return df


def read_manifest(manifest_file_pth):
    """
    Takes weird geotop output to establish driving data, model and sitename.
    :param manifest_file_pth: a string that is the folder name
    :return: [model, driving data, site]
    """
    df = pd.read_csv(manifest_file_pth, usecols=['site', 'model', 'forcing', 'parameters'])
    df.parameters = [line.replace('/', '.').split('.')[-2] for line in df.parameters]  # '.../*/Peat.inputs --> 'Peat'
    df.forcing = [line.split('_')[1] for line in df.forcing]  # 'scaled_merra2_1h_scf1.5_kdiBoreholes' --> 'merra2'
    df['model'] = df[['model', 'forcing', 'parameters']].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
    df = df.drop(['forcing', 'parameters'], axis=1)

    return df


def plot_mon_jra(m_nc, o_nc):
    # Get temp
    m_temp = m_nc["Tg"]
    o_temp = o_nc["soil_temperature"]

    # Get time
    m_time = m_nc["Date"]
    o_time = o_nc["time"]

    ref = date2num(
        datetime(1970, 1, 1, 0, 0, 0), units=m_time.units, calendar="standard"
    )

    start, end = o_time[0], o_time[-1]
    m_time = m_time - ref
    time_select = np.logical_and(m_time[:] > start, m_time[:] < end)

    fig, ax = plt.subplots()

    for site_index in range(12):
        data = m_temp[site_index, time_select, 1]
        ax.plot(mdates.num2date(m_time[time_select]), data[:], label='Model')
        ax.plot(mdates.num2date(o_time[:]), o_temp[site_index], label='Observations')
        ax.title('')

        # Create two subplots and unpack the output array immediately
        f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
        ax1.set_title('Sharing Y axis')

    plt.legend()
    plt.show()











