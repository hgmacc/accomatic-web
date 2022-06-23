import shutil
import sys
import typing
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset, date2num, num2date

"""
VAR NAMING CONVENTIONS
*_file = file_path (str)
*_nc = <netCDF4._netCDF4.Dataset>
prefix = m(model), o(obs), a(acco)
"""


def round_time(dt=None, roundTo=60):
    if dt == None:
        dt = datetime.now()
    seconds = (dt.replace(tzinfo=None) - dt.min).seconds
    rounding = (seconds + roundTo / 2) // roundTo * roundTo
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


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

    o_time = o_time + ref
    start, end = o_time[0], o_time[-1]

    time_select = np.logical_and(m_time[:] > start, m_time[:] < end)
    data = m_temp[site_index, time_select, 1]
    m_time = m_time[time_select]

    mdates.set_epoch("0001-01-01T00:00")

    plt.plot(mdates.num2date(m_time), data[:])
    plt.plot(mdates.num2date(o_time), o_temp[site_index, :])
    # plt.title("Geotop Simulation of Rock at %s", o_nc['simulations'][site_index, :])
    plt.legend()
    plt.savefig("output_plot.pdf")


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

    # acco['sitename'][:] = m_nc['geotop']['sitename'][:]
    # acco['model'][:] = m_nc['geotop']['simulation'][:]
    acco.comment = "Created from Accomatic"
    acco.date_created = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    acco.obs_source = f"Obs data source: {exp.obs_pth}"
    acco.mod_source = f"Model data source: {exp.model_pth}"

    acco.close()


def run_acco(m_nc, o_nc):
    """
    This function runs accomatic processes and populated acco nc group accordingly.
    :param m_nc: NC files with model groups and acco group
    :param o_nc: observational data for comparison
    :return: NA: Populates acco group of ncfile
    """
    # what models are in m_nc
    model_count = m_nc.groups
    # add obs in netcdf
    # add a group called meta = what is the raw data this is based on?
    # Meta = point to model and obs
    # terrain is an attribute of terrain
    # Terrain = everyone has their own ideas
    # Additional table to describe the site
    # Terrain types are not defined by
