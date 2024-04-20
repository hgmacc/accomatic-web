import os
import sys
import pickle

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import StrMethodFormatter

import pandas as pd
from box import get_model, get_colour


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"
sys.path.append("/home/hma000/accomatic-web/accomatic/")
from Experiment import *

pth = "data/pickles/14Mar_0.1_0.pickle"

with open(pth, "rb") as f_gst:
    exp = pickle.load(f_gst)

stat = "MAE"

df = exp.results
idx = pd.IndexSlice
df = exp.results.loc[idx[["res"], :, :, stat]].droplevel("mode")
data = {}
for mod in df.columns:
    name = get_model[mod]
    data[name] = np.concatenate([cell.arr for cell in df[mod]], axis=0)


x = list(range(100, 1335, 1))
x = [i / 100 for i in x]
x = [2**i for i in x]
x[-1] = 10000

for mod in data.keys():
    y = [np.mean(data[mod][: round(i)]) for i in x]
    plt.plot(x[:1220], y[:1220], label=mod, c=get_colour[mod])

plt.xscale("log")

# plt.legend(loc="upper right", fontsize="x-small")
plt.savefig(f"/home/hma000/accomatic-web/plotting/out/bootstrap{stat}.png")
