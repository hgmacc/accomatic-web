import glob
import os
import re
import sys

from Experiment import *
from NcReader import *
from Stats import *
from plotting.tables import *
from plotting.spiderline import spider
from plotting.box import one_exp
from plotting.heatmap import heat


sys.exit()
# RE RUN with longer spinup sims + new SCF + new AGG lol

plots = True
pth = "data/pickles/06Feb_0.1_0.pickle"

with open(pth, "rb") as f_gst:
    exp = pickle.load(f_gst)

f = open("data/csvs/06FEB.txt", "w+")

f.write("\nOverall: Mean performance")
df = get_overall_values(exp)
for model in ["era5", "jra55", "merra2", "ens"]:
    f.write(f"\n {get_model[model]}")
    f.write(f"\nmae: {mean_absolute_error(df[model], df.obs)}")
    f.write(f"\nbias: {bias(df[model], df.obs)}")
    f.write(f"\nr: {r_score(df[model], df.obs)}")

f.write("\n\nOverall: Rank Distribution")
heat(exp)
plt.close("all")

f.write("\n\nStats: Rank Distribution")
for stat in exp.stat_list:
    heat(exp, stat=stat)

if plots:
    f.write("\n\nStats: Boxplot")
    for s in exp.stat_list:
        one_exp(exp, stat=s, save=True)

f.write("\n\nTerrain: Rank Distribution")
for terr in set(exp.terr_list):
    heat(exp, terr=terr)
    plt.close("all")

    for stat in exp.stat_list:
        heat(exp, terr=terr, stat=stat)
        plt.close("all")

if plots:
    f.write("\n\nTerrain: Boxplot")
    for s in exp.stat_list:
        for t in set(exp.terr_list):
            one_exp(exp, stat=s, terr=t, save=True)
            plt.close("all")

    f.write("\n\nSpider plot (perhaps, with confidence intervals....)")
    spider(exp)
    plt.close("all")


f.close()

pth = "data/pickles/06Feb_0.1_0.pickle"

with open(pth, "rb") as f_gst:
    exp = pickle.load(f_gst)
