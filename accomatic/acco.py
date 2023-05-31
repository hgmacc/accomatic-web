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

# python accomatic/acco.py -f /home/hma000/accomatic-web/tests/test_data/toml/MAR_NWT.toml

if __name__ == "__main__":
    # arg_input = get_toml_pth(sys.argv)
    # e = Experiment(arg_input)
    # a = e.obs()
    # a = a.groupby('sitename').mean()
    # print(a.loc[['NGO-DD-2023']])
    
    # sys.exit()
    
    # build(e)  
    # a = e.results.groupby(['sim']).mean()
    # from Plotting import seasonal_violin
    # seasonal_violin(e)
    # a = a.groupby(['szn']).rank(method="max").astype(int)
    
    # print(e.results.groupby('site').RMSE.mean())
    