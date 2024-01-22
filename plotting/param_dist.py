import sys
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
import seaborn as sns
from matplotlib import patches
from matplotlib.patches import Patch

warnings.simplefilter(action="ignore")


def getval(dat, lon, lat):
    """
    Gets value from tif data. Must first convert lat/lon coordinate to 3413 CRS.

    Args:
        dat (rasterio data): array from tif file
        lon (float): longitude coordinate
        lat (float): latitude coordinate

    Returns:
        float: Value in dat at lat/lon coordinate.
    """
    z = dat.read()[0]
    to_utm = pyproj.Transformer.from_crs(4326, 3413, always_xy=True)
    a, b = to_utm.transform(lon, lat)
    idx = dat.index(a, b)

    try:
        res = z[idx]
    except IndexError:
        res = np.nan
    return res


def populate_df_col(place, df):
    """
    Iterates through .tif files for a given location and getval() for each site.

    Args:
        place (string): Yellowknife, Lac de Gras or KDI
        abbrev (string): The abbrev at the beginning of sitenames to subset coordinate df

    Returns:
        pd.DataFrame: a dataframe with twi, con and iso columns added
    """
    to_do = ["elevation", "twi", "con", "slope", "aspect"]

    for thing in to_do:
        path = f"/home/hma000/storage/terrain_exp/dem_processed/{place}/{place}_dem_10m_{thing}.tif"
        dat = rio.open(path)

        a = df.apply(lambda x: getval(dat, float(x["lon"]), float(x["lat"])), axis=1)
        try:
            df[thing] = a
        except ValueError:
            print(a)

        dat.close()

    return df


def build_terrain_csv():
    """
    For each location, build df then merge them all at the end and write data to a csv file.
    """
    coords = pd.read_csv(
        "/home/hma000/storage/terrain_exp/ykl_coords.csv",
        usecols=["name", "lat", "lon", "elevation_in_metres", "sky_view"],
    )

    places = ["ldg", "kdi", "yk"]

    coords["clust"] = coords["name"].str[:2]
    coords.clust.replace(["RO", "Bu"], "NG", inplace=True)
    coords.clust.replace(["NG"], "LD", inplace=True)

    l_df = []
    for place in places:
        l_df.append(
            populate_df_col(place, coords[coords["clust"] == place[:2].upper()])
        )
    df = pd.concat(l_df)

    df.to_csv(
        "/home/hma000/storage/terrain_exp/ykl_terrain.csv",
        columns=[
            "name",
            "clust",
            "elevation_in_metres",
            "elevation",
            "twi",
            "con",
            "slope",
            "aspect",
        ],
    )
    return True


def normalize_data(data):
    for col in ["twi", "con"]:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data


def sort_data(data):
    param_dict = {
        "twi": {"bins": [0, 0.33, 0.66, 1.0], "labels": ["0", "1", "2"]},
        "con": {
            "bins": [0, 0.25, 0.50, 0.75, 1.0],
            "labels": ["3", "2", "1", "0"],
        },
        "aspect": {"bins": [0, 45, 270, 360], "labels": ["0", "1", "0"]},
        "slope": {"bins": [10, 20, 70], "labels": ["0", "1"]},
    }
    for par in param_dict.keys():
        df[par] = pd.cut(
            df[par],
            bins=param_dict[par]["bins"],
            labels=param_dict[par]["labels"],
            ordered=False,
        )
    return data


df = pd.read_csv(
    "/home/hma000/storage/terrain_exp/ykl_coords.csv",
    usecols=["name", "lat", "lon", "elevation_in_metres", "sky_view"],
)
populate_df_col("yk", df)
