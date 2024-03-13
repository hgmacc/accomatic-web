# NOT FOR PLOTTING
# This produced dataframe data that is manualy entered into latex thesis document.
import sys
from matplotlib.dates import num2date

sys.path.append("/home/hma000/accomatic-web/accomatic/")
import pickle
from Stats import *
from Experiment import *
import pandas as pd
from plotting.box import get_model
import matplotlib.pyplot as plt

sites_list = [
    "NGO-DD-1009",
    "NGO-DD-1005",
    "KDI-W-Org2",
    "KDI-E-Org2",
    "KDI-E-Wet",
    "YK16-SO07",
    "NGO-DD-2020",
    "NGO-DD-2023",
    "NGO-DD-2008",
    "NGO-DD-2029",
    "KDI-W-Ttop",
    "KDI-E-Ttop",
    "NGO-DD-1012",
    "NGO-DD-2004",
    "NGO-DD-2033",
    "NGO-DD-2019",
    "NGO-DD-1011",
    "NGO-DD-2034",
    # "KDI-W-Snowdrift",
    "KDI-R1",
    "KDI-R2",
    "KDI-R3",
    "YK16-RH02",
    "YK16-RH03",
    "YK16-RH01",
    "ROCKT1",
    "ROCKT2",
    "ROCKT3",
]
pth = "/home/hma000/accomatic-web/snow/"
reanalyses = ["merra2", "era5", "jra55"]


def plot_snow():
    fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(10, 10))
    i = 0

    df = read_snow(f"{pth}dat/result_snow_depth.nc")
    df.reset_index(inplace=True)
    df["clust"] = df.sitename.str[:2]
    df = df.loc[df.sitename.isin(sites_list)]

    df["time"] = pd.to_datetime(df.time)
    df.set_index("time", inplace=True)

    for dat in reanalyses:
        for cluster, c in zip(
            ["YK", "NG", "KD"],
            ["r", "y", "b"],
        ):
            a = df[df.clust == cluster][dat]
            if dat == "merra2" and cluster == "KD":
                break
            ax[i].plot(a.resample("m").mean(), label=cluster, linestyle=":", c=c)
            # ax[i].fill_between(
            #     range(12),
            #     a.resample("m").min(),
            #     a.resample("m").max(),
            #     alpha=0.3,
            #     color=c,
            # )
            if cluster == "YK":
                ax[i].text(x=100, y=400, s=dat.upper())
                ax[1].legend(loc="upper right")
                ax[1].set_ylabel("Snow depth (mm)")
        i = i + 1
    ax[0].set_xticks([])
    ax[1].set_xticks([])

    plt.savefig(f"{pth}out/snow.png")


def plot_merra2_snow():
    fig = plt.plot(figsize=(6, 8))
    df = read_snow(f"{pth}dat/result_snow_depth.nc")

    l = [
        "KDI-W-Org2",
        "KDI-E-Org2",
        "KDI-E-Wet",
        "KDI-W-Ttop",
        "KDI-E-Ttop",
        "KDI-R1",
        "KDI-R2",
        "KDI-R3",
    ]
    df.reset_index(inplace=True)
    df = df.loc[df.sitename.isin(l)][["time", "sitename", "merra2"]]

    sns.lineplot(data=df, x="time", y="merra2", hue="sitename", legend=True)
    plt.savefig(f"{pth}out/merra2_snow.png")
    plt.clf()


def plot_from_surfacetxt():
    fig, ax = plt.subplots(figsize=(20, 10))
    for data in reanalyses:
        df = pd.read_csv(f"{pth}dat/yk_{data}_forcing.txt")
        df.set_index(
            pd.to_datetime(df["Date"], format="%d/%m/%Y %H:%M"),
            inplace=True,
        )
        df = df[df.index.year.isin([2015, 2016, 2017, 2018, 2019, 2020])]
        # Hourly sum of rain
        prec = df.IPrec.resample("M").sum()
        plt.plot(prec, label=data)
    plt.ylabel("Annual Precipitation (mm)", labelpad=20)
    plt.legend()
    plt.savefig(f"{pth}/out/YK16SO07_monthly_precip.png")


scaled_pth = "/home/hma000/storage/yk_kdi_ldg/scaled/ncfiles/"


def plot_from_scaled():
    fig, ax = plt.subplots(nrows=3, ncols=1, sharey=True, figsize=(10, 10))
    i = 0
    for dat in reanalyses:
        nc = xr.open_mfdataset(
            f"{scaled_pth}/scaled_{dat}_1h_2015.nc", engine="netcdf4"
        )

        df = nc["station_name"].to_dataframe().reset_index()
        print(len(df.station_name.unique()))
        df.station_name = [line.decode("utf-8") for line in df.station_name]
        df = df.loc[df.station_name.isin(sites_list)]
        print(df.head())
        sys.exit()

        df.station_name = df.station_name[:2]

        for cluster, c in zip(["YK", "KD", "NG"], ["r", "b", "y"]):
            stns = df[df.station_name == cluster].station.to_list()
            a = nc.PREC_sur.isel(station=stns).groupby("time.month").sum() * 3600

            ax[i].plot(np.mean(a, axis=1), label=cluster, linestyle=":", c=c)
            ax[i].fill_between(
                range(12),
                np.min(a, axis=1),
                np.max(a, axis=1),
                alpha=0.3,
                color=c,
            )

            if cluster == "NG":
                ax[i].text(x=0, y=100, s=dat.upper())
                if i == 0:
                    ax[0].legend(loc="upper right")
                if i == 1:
                    ax[1].set_ylabel("Precipitation (mm / month)")

        ax[i].set_xticks([])
        i = i + 1
        nc.close()

    ax[2].set_xticks(range(0, 12, 2), ["JAN", "MAR", "MAY", "JUL", "SEP", "NOV"])
    plt.savefig(f"{pth}out/precip.png")
    plt.clf()


# plot_snow()
plot_from_scaled()
plot_merra2_snow()
