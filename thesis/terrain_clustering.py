import warnings
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
from rasterio.windows import from_bounds

from matplotlib import patches
from collections import Counter
import seaborn as sns

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

    to_do = ["twi", "con", "iso"]
    for thing in to_do:
        path = f"/home/hma000/storage/terrain_exp/dem_processed/{place}/{place}_dem_10m_{thing}.tif"
        dat = rio.open(path)

        df[thing] = df.apply(
            lambda x: getval(dat, float(x["lon"]), float(x["lat"])), axis=1
        )
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

    places = ["yk", "kdi", "ldg"]
    abbrevs = ["YK", "KD", "NG"]

    coords["clust"] = coords["name"].str[:2]
    coords.clust.replace(["RO", "Bu"], "NG", inplace=True)

    l_df = []
    for place, abbrev in zip(places, abbrevs):
        l_df.append(populate_df_col(place, coords[coords["clust"] == abbrev]))
    df = pd.concat(l_df)

    df.to_csv(
        "/home/hma000/storage/terrain_exp/ykl_terrain.csv",
        columns=["name", "clust", "elevation_in_metres", "twi", "con", "iso"],
    )
    return True


def normalize_data(data):
    for col in ["twi", "con", "iso"]:
        data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())
    return data


def sort_data(data):
    param_dict = {
        "twi": {0.33: "Dry", 0.66: "Mix", 1.0: "Wet"},
        "con": {0.25: "Low", 0.50: "Normal", 0.75: "More", 1.0: "Drift"},
        "iso": {0.33: "Low", 0.66: "Mix", 1.0: "High"},
    }

    for param in param_dict.keys():
        new_data = []
        for row in data[param]:
            for value in param_dict[param].keys():
                if row <= value:
                    new_data.append(param_dict[param][value])
                    break
        data[f"{param}_str"] = new_data

    return param_dict, data


def plot_param_distribution():
    df = pd.read_csv(
        "/home/hma000/storage/terrain_exp/ykl_terrain.csv",
        usecols=["name", "twi", "con", "iso", "clust"],
    )

    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, figsize=(10, 4))
    df = df.dropna()
    df = normalize_data(df)

    colours = ["#D09862", "#7ABFD5", "#B1AEA1"]
    clust_colours = {"YK": "#1CE1CE", "KD": "#F3700E", "NG": "#F50B00"}
    param_list = ["con", "twi", "iso"]
    titles = ["Snow Collection", "Terrain Wetness", "Incoming Solar Radiation"]
    for i in range(3):
        c = colours[i]
        param = param_list[i]
        plt.subplot(2, 3, i + 1)
        for clust in df.clust.unique():
            plt.hist(
                df[df.clust == clust][param],
                histtype="step",
                label=clust,
                color=c,
                ec=clust_colours[clust],
            )
        plt.title(titles[i])
        if i == 2:
            plt.legend()

    param_dict, df = sort_data(df)
    for i in range(3):
        plt.subplot(2, 3, i + 4)

        # i.e. groups = ["dry", "mix", "wet"]
        groups = param_dict[param_list[i]].values()

        param = f"{param_list[i]}_str"
        # data = {'Normal': 51, 'More': 14, 'Low': 14, 'Drift': 3}
        data = Counter(df[param])
        data = [data[i] for i in groups]
        plt.bar(groups, data, color=colours[i])

    plt.savefig("plot_param_distribution.png")


def plot_param_distribution():
    df = pd.read_csv(
        "/home/hma000/storage/terrain_exp/ykl_terrain.csv",
        usecols=["name", "twi", "con", "iso", "clust"],
    )

    fig, axs = plt.subplots(nrows=2, ncols=3, sharex=False, figsize=(10, 4))
    df = df.dropna()
    df = normalize_data(df)

    colours = ["#B1AEA1", "#7ABFD5", "#D09862"]
    clust_colours = {"YK": "#1CE1CE", "KD": "#F3700E", "NG": "#F50B00"}
    param_list = ["con", "twi", "iso"]
    titles = ["Snow Collection", "Terrain Wetness", "Incoming Solar Radiation"]
    for i in range(3):
        c = colours[i]
        param = param_list[i]
        plt.subplot(2, 3, i + 1)
        for clust in df.clust.unique():
            plt.hist(
                df[df.clust == clust][param],
                histtype="step",
                label=clust,
                color=c,
                ec=clust_colours[clust],
            )
        plt.title(titles[i])
        if i == 2:
            plt.legend()

    param_dict, df = sort_data(df)
    for i in range(3):
        plt.subplot(2, 3, i + 4)

        # i.e. groups = ["dry", "mix", "wet"]
        groups = param_dict[param_list[i]].values()

        param = f"{param_list[i]}_str"
        # data = {'Normal': 51, 'More': 14, 'Low': 14, 'Drift': 3}
        data = Counter(df[param])
        data = [data[i] for i in groups]
        plt.bar(groups, data, color=colours[i])

    plt.savefig("plot_param_distribution.png")


def plot_dem_distribution_merge():
    fig, axs = plt.subplots(nrows=3, ncols=1, sharex=True, sharey=False, figsize=(6, 6))
    things = ["twi", "con", "iso"]
    clusters = ["yk", "kdi", "ldg"]

    colours = ["#B1AEA1", "#7ABFD5", "#D09862"]
    clust_colours = {"YK": "#1CE1CE", "KD": "#F3700E", "NG": "#F50B00"}

    df = pd.read_csv(
        "/home/hma000/storage/terrain_exp/ykl_terrain.csv",
        usecols=["name", "twi", "con", "iso", "clust"],
    )

    index = 1
    for par_i in range(3):

        plt.subplot(3, 1, index)
        index = index + 1

        data_l = []
        for pth in range(3):
            path = f"/home/hma000/storage/terrain_exp/dem_processed/{clusters[pth]}/{clusters[pth]}_dem_10m_{things[par_i]}.tif"

            dat = rio.open(path).read().flatten()
            data_l.append(dat[dat > 0])
        dat = np.concatenate(data_l, axis=0)

        amin, amax = min(dat), max(dat)
        for n, val in enumerate(dat):
            dat[n] = (val - amin) / (amax - amin)

        # Normalizing obs to entire dataset
        arr = [(val - amin) / (amax - amin) for val in df[things[par_i]]]

        plt.hist(
            arr,
            label="Obs",
            histtype="step",
            density=1,
            color="red",
            linewidth=2,
        )

        plt.hist(
            dat,
            label=things[par_i],
            density=1,
            color=colours[par_i],
            alpha=0.5,
        )

        plt.legend(frameon=False)
        plt.title(things[par_i])

    plt.savefig("plot_dem_distribution_merge.png")


def plot_dem_distribution_seperate():
    # Creats 9 subplots (3x3)

    fig, axs = plt.subplots(
        nrows=3, ncols=3, sharex=True, sharey=False, figsize=(12, 7)
    )

    things = ["twi", "con", "iso"]
    clusters = ["yk", "kdi", "ldg"]

    par_colours = {"twi": "#B1AEA1", "con": "#7ABFD5", "iso": "#D09862"}
    clust_colours = {"YK": "#1CE1CE", "KD": "#F3700E", "NG": "#F50B00"}

    df = pd.read_csv(
        "/home/hma000/storage/terrain_exp/ykl_terrain.csv",
        usecols=["name", "twi", "con", "iso", "clust"],
    )

    cluster_list = df.clust.unique()
    index = 1
    for par_i in range(3):
        for clust_i in range(3):

            plt.subplot(3, 3, index)
            index = index + 1

            path = f"/home/hma000/storage/terrain_exp/dem_processed/{clusters[clust_i]}/{clusters[clust_i]}_dem_10m_{things[par_i]}.tif"

            dat = rio.open(path).read().flatten()
            dat = dat[dat > 0]

            amin, amax = min(dat), max(dat)
            for n, val in enumerate(dat):
                dat[n] = (val - amin) / (amax - amin)

            # Normalizing obs to entire dataset
            arr = [
                (val - amin) / (amax - amin)
                for val in df[df.clust == cluster_list[clust_i][:2].upper()][
                    things[par_i]
                ]
            ]

            lab = {"obs": "_Observed_", "par": "_DEM Result_"}
            if index == 0:
                [lab[val].strip("_") for val in lab.keys()]

            plt.hist(
                arr,
                label="Obs",
                histtype="step",
                # bins=8,
                density=1,
                color=clust_colours[cluster_list[clust_i]],
                linewidth=2,
            )

            plt.hist(
                dat,
                # bins=8,
                label="DEM",
                density=1,
                color=par_colours[things[par_i]],
                alpha=0.5,
            )

            if index in [2, 6, 10]:
                plt.legend(frameon=False)

            if clust_i == 0:
                plt.ylabel(things[par_i].upper())

            if index < 5:
                plt.title(cluster_list[clust_i])

    plt.savefig("plot_dem_distribution.png")


plot_dem_distribution_seperate()
