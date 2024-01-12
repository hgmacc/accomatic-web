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


# python accomatic/acco.py -f /home/hma000/accomatic-web/data/toml/test.toml

if __name__ == "__main__":
    arg_input = get_toml_pth(sys.argv)
    # exp = Experiment(arg_input)
    pth = "/home/hma000/accomatic-web/data/pickles/2024-01-12_results.pickle"
    print(pth)
    with open(pth, "rb") as f_gst:
        exp = pickle.load(f_gst)

    from plotting.heatmap import *

    for terr in exp.terr_list:
        terr_desc = [
            "PEATLAND",
            "COURSE_HILLTOP",
            "FINE_HILLTOP",
            "SNOWDRIFT",
            "HOR_ROCK",
        ]
        exp.rank_dist = rank_distribution(exp, terr=terr)
        bias_heatmap(exp, save=True, title=f"heat/{terr_desc[terr - 1]}_bias")
        rank_dist_heatmap(exp.rank_dist, save=True, title=f"heat/{terr_desc[terr - 1]}")
