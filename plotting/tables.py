# NOT FOR PLOTTING
# This produced dataframe data that is manualy entered into latex thesis document.
import sys

sys.path.append("/home/hma000/accomatic-web/accomatic/")
import pickle
from Stats import *
from Experiment import *
import pandas as pd
from plotting.box import get_model


def terrain_std_var(exp):

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


def terrain_rank_distribution(exp):
    for terrain in set(exp.terr_list):
        df = rank_distribution(exp, terr=terrain)
        # df = bias_distribution(exp, terr=terrain)
        print(terrain)
        print(df.round(2))


# import os

# for root, dirs, files in os.walk(
#     "/home/hma000/accomatic-web/data/pickles/", topdown=False
# ):
#     for name in files:
#         pth = os.path.join(root, name)
#         print(pth)
#         with open(pth, "rb") as f_gst:
#             exp = pickle.load(f_gst)
#             df = rank_distribution(exp)
#             print(df.round(2))lk
#             f_gst.close()


def get_overall_values(exp):
    """
    TALK ABOUT BIAS!! WOOOOW
    """

    o = exp.obs()
    o["t"] = o.index.get_level_values(1) + "-" + o.index.get_level_values(0).astype(str)
    o.reset_index(drop=True, inplace=True)
    o.set_index("t", inplace=True)

    m = exp.mod()
    m["t"] = m.index.get_level_values(1) + "-" + m.index.get_level_values(0).astype(str)
    m.reset_index(drop=True, inplace=True)
    m.set_index("t", inplace=True)

    df = exp.obs().join(exp.mod()).dropna()
    return df
