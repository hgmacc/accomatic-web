import getopt
import glob
import os
import re
import sys

from Experiment import *
from NcReader import *
from Stats import *
from Plotting import *
from thesis.spiderplot import spiderplot


def get_toml_pth(argv):
    arg_input = ""
    arg_help = "{0} -f <toml_file_path>".format(argv[0])

    try:
        opts, args = getopt.getopt(argv[1:], "h:f:", ["help", "file="])
    except:
        print(arg_help)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(arg_help)
            sys.exit(2)
        elif opt in ("-f", "--file"):
            if not os.path.exists(arg):
                print(f"ERROR: Path '{arg}' does not exist.")
                sys.exit()
            else:
                arg_input = arg

    return arg_input


# python accomatic/acco.py -f /home/hma000/accomatic-web/tests/test_data/toml/MAR_NWT.toml

if __name__ == "__main__":
    arg_input = get_toml_pth(sys.argv)
    import netCDF4 as nc

    f2 = nc.Dataset("/home/hma000/storage/yk_kdi_ldg/scaled/scaled_era5_1h_1980.nc")
    names_out = nc.stringtochar(np.array(f2["station_name"]))

    for i in ["merra2", "jra55"]:
        f = nc.Dataset(
            f"/fs/yedoma/data/globsim/YK-KDI-LDG_scaled_for_geotop/scaled_{i}_1h.nc",
            "w",
        )
        print(f"{i} is open.")
        f.createDimension("name_strlen", 32)
        f.createVariable("station_name", "S1", ("station", "name_strlen"))
        f.standard_name = "platform_name"
        f.units = ""
        f[:] = names_out

        f.close()
        print(f"{i} is complete. ")
    f2.close()

    # exp = Experiment(arg_input)

    # cluster_timeseries(exp)
    sys.exit()
    all_o = o
    all_o.index = all_o.level_0
    plt.subplot(211)
    sns.lineplot(
        data=all_o.dropna(),
        x="level_0",
        y="obs",
        palette=get_color_gradient(
            c1="#804203", c2="#ffb161", n=len(all_o.sitename.unique())
        ),
        hue="sitename",
        legend=False,
        linewidth=0.5,
    )

    plt.xlabel("")
    plt.ylabel("GST ˚C")

    build(exp)
    csv_rank(exp)
    spiderplot(exp)
