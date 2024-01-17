import sys
import pickle

sys.path.append("../")
from accomatic.Experiment import *
from accomatic.Stats import *
from plotting.box import *


pth = "/home/hma000/accomatic-web/data/pickles/NOV16_bs1000_d05.pickle"
with open(pth, "rb") as f:
    exp = pickle.load(f)

boxplot(exp, stat="MAE", save=True)

sys.exit()
ax[1].plot(boxplot(exp, stat="MAE", save=False))
ax[2].plot(boxplot(exp, stat="MAE", save=False))


plt.savefig("/home/hma000/accomatic-web/plotting/out/test.png")


sys.exit()
plt.subplot(131)
ax1 = boxplot(exp, stat="MAE", save=False)
ax1.remove()
plt.plot()

plt.subplot(132)
ax2 = boxplot(exp, stat="dr", save=False)
plt.plot()

plt.subplot(133)
ax3 = boxplot(exp, stat="BIAS", save=False)
plt.plot()

plt.savefig("/home/hma000/accomatic-web/plotting/out/test.png")
# For stat in statlist, create violin plot
