# Can we visualize that an increase in data points = more certainty in rankings at the granular-most (terr-szn) level? 
import os
import sys
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

import pandas as pd

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"

import sys
sys.path.append('/home/hma000/accomatic-web/accomatic/')
from Experiment import *

exp = read_exp("data/pickles/09May_0.1_0.pickle")
print(exp)


print(exp.stat_list, set(exp.terr_list))

# idx = pd.IndexSlice
# df = exp.results.loc[idx[["res"], terr, szn, stat]].droplevel("mode")


