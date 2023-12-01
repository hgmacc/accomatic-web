import xarray as xr
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import netCDF4

# For each reanalysis dataset, plot:
# For each variable:
# Mean of all sites + std deviation

file_pth = "/home/hma000/storage/yk_kdi_ldg/scaled/ncfiles/"


def fixing_stn_name():
    good_file = (
        "/home/hma000/storage/yk_kdi_ldg/scaled/ncfiles/scaled_merra2_1h_1995.nc"
    )
    to_fix_file = (
        "/home/hma000/storage/yk_kdi_ldg/scaled/ncfiles/scaled_merra2_1h_1985_old.nc"
    )

    f2 = xr.open_mfdataset(good_file, engine="netcdf4")
    f = xr.open_mfdataset(to_fix_file, engine="netcdf4")

    f["station_name"] = f2["station_name"]
    f.to_netcdf(
        path=f"/home/hma000/storage/yk_kdi_ldg/scaled/ncfiles/scaled_merra2_1h_1985.nc",
        mode="w",
    )

    f.close()
    f2.close()


vardict = {
    "PRESS_pl": "Air Pressure Pa",
    "SH_sur": "Specifc Humidity",
    "AIRT_sur": "2m Air Temp. ˚C",
    "PREC_sur": "Precipitation $mm s^{-1}$",
    "SW_sur": "Shortwave Rad. $W m^{-2}$",
    "LW_sur": "Longwave Rad. $W m^{-2}$",
}


def plotting_meteo(year, save=False):

    fig, axs = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(15, 6))
    for dataset, col in zip(
        ["era5", "jra55", "merra2"], ["#008080", "#f50b00", "#f3700e"]
    ):
        f = xr.open_mfdataset(
            f"{file_pth}scaled_{dataset}_1h_{year}.nc", engine="netcdf4"
        )
        vars = [
            "station_name",
            "PRESS_pl",
            "SH_sur",
            "AIRT_sur",
            "PREC_sur",
            "SW_sur",
            "LW_sur",
        ]
        try:
            df = f[vars].to_dataframe()
        except KeyError:
            print(dataset, year)
            sys.exit()
        df.station_name = [line.decode("utf-8") for line in df.station_name]
        df = df.reset_index().drop(columns="station")
        df["time"] = pd.to_datetime(df["time"]).dt.date
        df = df.groupby(["time", "station_name"]).mean().reset_index()

        plot_index = 1
        for var in vars[1:]:
            plt.subplot(2, 3, plot_index)
            plot_index = plot_index + 1
            sns.lineplot(
                data=df[["time", "station_name", var]],
                x="time",
                y=var,
                color=col,
                # err_style="bars",
                # errorbar=("se", 2),
                linewidth=1,
                legend=False,
            )
            plt.xlabel("")
            plt.ylabel(vardict[var])
            if dataset == "era5":
                locs, labels = plt.xticks()
                plt.xticks(locs[::2], labels[::2])

    plt.suptitle(year)
    plt.tight_layout(pad=1.0)
    plt.savefig(f"/home/hma000/accomatic-web/plotting/out/meteo_{year}.png")
    plt.clf()
    plt.close(fig)


def plotting_precip():
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
    first_plt_legend = True

    plot_index = 1
    for year in [1985, 1995, 2005, 2015]:
        plt.subplot(2, 2, plot_index)
        plot_index = plot_index + 1

        f = xr.open_mfdataset(f"{file_pth}scaled_jra55_1h_{year}.nc", engine="netcdf4")
        vars = ["station_name", "PREC_sur"]
        df = f[vars].to_dataframe()
        df.station_name = [line.decode("utf-8")[:2] for line in df.station_name]
        df.station_name.replace(["RO", "Bu"], "NG", inplace=True)
        df = df.reset_index().drop(columns="station")
        df["time"] = pd.to_datetime(df["time"]).dt.date
        df = df.groupby(["time", "station_name"]).mean().reset_index()
        df = df[df.station_name.isin(["KD", "NG", "YK"])]
        for cluster, col in zip(
            df.station_name.unique(), ["#008080", "#f50b00", "#f3700e"]
        ):
            sns.lineplot(
                data=df[df.station_name == cluster],
                x="time",
                y="PREC_sur",
                color=col,
                err_style="bars",
                errorbar=("se", 2),
                label=cluster,
                linewidth=1,
                legend=first_plt_legend,
            )
        plt.title(year)
        plt.xlabel("")
        first_plt_legend = False
        locs, labels = plt.xticks()
        plt.xticks(locs[::2], labels[::2])

    plt.savefig(f"jra55_precip.png")
    plt.clf()
    plt.close(fig)


plotting_meteo(1985)
