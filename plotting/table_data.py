# NOT FOR PLOTTING
# This produced dataframe data that is manualy entered into latex thesis document.
import sys

sys.add_path("../")
import pickle
from accomatic.Stats import *


def terrain_std_var():

    pth = "/home/hma000/accomatic-web/data/pickles/2024-01-15_results.pickle"
    with open(pth, "rb") as f_gst:
        exp = pickle.load(f_gst)
    o = exp.obs()
    o["terr"] = [exp.terr_dict()[x] for x in o.index.get_level_values(1)]
    idx = pd.IndexSlice

    df = exp.results
    idx = pd.IndexSlice
    for terrain in set(exp.terr_list):
        print(f"\n\n{terrain}")

        otmp = o[o.terr == terrain]
        print(otmp.obs.mean())
        print(otmp.obs.std())

        tmp = df.loc[idx[["res"], terrain, :, "WILL"]].droplevel("mode")
        tmp = tmp.apply(average_data)
        print(tmp)


def terrain_rank_distribution():
    pth = "/home/hma000/accomatic-web/data/pickles/2024-01-15_results.pickle"
    with open(pth, "rb") as f_gst:
        exp = pickle.load(f_gst)

    for terrain in set(exp.terr_list):
        df = rank_distribution(exp, terr=terrain)
        print(df.round(2).to_latex())
