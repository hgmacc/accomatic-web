import sys
import os
import getopt
from netCDF4 import Dataset

from Settings import *
from NcReader import *
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
            print(arg_help)  # print the help message
            sys.exit(2)
        elif opt in ("-f", "--file"):  # get config toml file path
            if not os.path.exists(arg):
                print(f"ERROR: Path '{arg}' does not exist.")
                sys.exit()
            else: arg_input = arg
        
    return arg_input
            

if __name__ == "__main__":
    arg_input = get_toml_pth(sys.argv)
    e = Experiment(arg_input)
    






