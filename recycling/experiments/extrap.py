import sys
import pickle

sys.path.append("../../")
from accomatic.Experiment import *
from accomatic.Stats import *
from plotting.heatmap import *

# exp = Experiment("/home/hma000/accomatic-web/data/toml/test.toml")
# build(exp)

exp50 = Experiment("/home/hma000/accomatic-web/data/toml/50cm.toml")
build(exp50)

# pth = "/home/hma000/accomatic-web/data/pickles/14NOV23_1000_gst.pickle"
# with open(pth, "rb") as f_gst:
#     exp = pickle.load(f_gst)

# pth = "/home/hma000/accomatic-web/data/pickles/50cm.pickle"
# with open(pth, "rb") as f_05:
#     exp_05 = pickle.load(f_05)

# dist_diff = exp.rank_dist - exp_05.rank_dist
# exp.rank_dist = dist_diff
# rank_dist_heatmap(exp)
