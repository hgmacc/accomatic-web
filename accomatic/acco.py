import sys
import os
import getopt
import re

from Experiment import *
from Stats import *

from Plotting import *


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


# python accomatic/acco.py -f /home/hma000/accomatic-web/tests/test_data/toml/SEP_KDI.toml

if __name__ == "__main__":
    arg_input = get_toml_pth(sys.argv)
    e = Experiment(arg_input)
    build(e)
    xy_tmp(e)


    # a = e.results.groupby(['szn', 'sim']).mean().drop(columns=['data_avail'])
    # a = a.groupby(['szn']).rank(method="max").astype(int)
    # print(e.results.groupby('site').RMSE.mean())


