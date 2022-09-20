import sys
import os
import getopt
import re
from netCDF4 import Dataset
from tsp.readers import read_gtpem

from Experiment import *


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


if __name__ == "__main__":
    arg_input = get_toml_pth(sys.argv)
    e = Experiment(arg_input)
    # Now run the stats. You're unsure as to whether you want to
    # create a new class for results, put the values into acco.nc
    # or just keep as a temp DF. I think writing to an .nc file might
    # make sense if there's lots of post-processing.
    # Then you could have "acco build" (produce acco.nc file)
    # OR "acco analyse" (examine acco.nc file for summary)
    # AND "acco visual" (produce high level pdf summary of findings)
