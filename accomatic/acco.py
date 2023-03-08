import getopt
import glob
import os
import re
import sys

from Experiment import *
from NcReader import *
from Stats import *

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


# python accomatic/acco.py -f /home/hma000/accomatic-web/tests/test_data/toml/NOV_NWT.toml

if __name__ == "__main__":
    arg_input = get_toml_pth(sys.argv)

    for file in glob.glob('/project/s/stgruber/hma000/talikForcing/ykl/*.nc'):
        f = xr.open_mfdataset(file)
        print(f.dims); sys.exit()
        time = f.time
        print('The time variable shape: %s and dimensions: %s' % (time.shape, time.dims))
    
    
    e = Experiment(arg_input)
    build(e) 
    #e.results.groupby(['sim', 'szn']).mean().drop(columns=['data_avail'])
    from Plotting import xy_site_plot
    for site in e.sites_list:
        xy_site_plot(e, site)

    # a = e.results.groupby(['sim', 'szn']).mean().drop(columns=['data_avail'])
    # a = a.groupby(['szn']).rank(method="max").astype(int)
    # print(e.results.groupby('site').RMSE.mean())
