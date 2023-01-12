import os
import sys
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio

warnings.simplefilter(action="ignore")

# to_utm = pyproj.Transformer.from_crs(4326, 3413, always_xy=True)
coords = pd.read_csv(
    "/home/hma000/storage/terrain_exp/ykl_coords.csv",
    usecols=["name", "lat", "lon", "elevation_in_metres", "sky_view"],
)


def getval(dat, lon, lat):
    z = dat.read()[0]
    to_utm = pyproj.Transformer.from_crs(4326, 3413, always_xy=True)
    a, b = to_utm.transform(lon, lat)
    idx = dat.index(a, b)
    return z[idx]


to_do = ["twi", "con", "clip"]
places = ["yk", "kdi", "ldg"]
abbrevs = ["YK", "KDI", "NGO"]


def run(place, abbrev):
    df = coords[coords["name"].str.contains(abbrev)]
    for thing in to_do:
        path = f"/home/hma000/storage/terrain_exp/dem_processed/{place}/{place}_dem_10m_{thing}.tif"
        dat = rio.open(path)
        df[thing] = df.apply(
            lambda x: getval(dat, float(x["lon"]), float(x["lat"])), axis=1
        )
        dat.close()
    return df


l_df = []
for place, abbrev in zip(places, abbrevs):
    l_df.append(run(place, abbrev))

df = pd.concat(l_df)
print(df.head())

df.to_csv(
    "/home/hma000/storage/terrain_exp/ykl_terrain.csv",
    columns=["name", "elevation_in_metres", "twi", "con", "clip"],
)
