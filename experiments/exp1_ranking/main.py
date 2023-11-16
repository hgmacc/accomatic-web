import sys
import pickle

sys.path.append("../../")
from accomatic.Experiment import *
from accomatic.Stats import *
from plotting.heatmap import *


exp = Experiment("/home/hma000/accomatic-web/data/toml/run.toml")

sys.exit()


####### NEW #############################################################
exp = Experiment("/home/hma000/accomatic-web/data/toml/run_05.toml")
build(exp)
pth = "/home/hma000/accomatic-web/data/pickles/14NOV23_1000_05.pickle"
with open(pth, "wb") as handle:
    pickle.dump(exp, handle, protocol=pickle.HIGHEST_PROTOCOL)
handle.close()

####### OLD #############################################################
pth = "/home/hma000/accomatic-web/data/pickles/14NOV23_1000_gst.pickle"
with open(pth, "rb") as f_gst:
    exp = pickle.load(f_gst)
