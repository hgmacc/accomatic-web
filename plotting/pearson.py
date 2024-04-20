from sinwave import sin

import numpy as np
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["font.size"] = "16"

days = 1800

x = np.array(range(days))
obs = sin(x, freq=360, noise=True, noise_val=1)
mod = sin(x, freq=360, noise=True, noise_val=1)


# How do these perform over entire year?
# How do these perform by month? Then averaged?

total = np.corrcoef(obs, mod)[0, 1]
print("Overall :", total)

monthly_indeces = [i for i in range(30, days, 30)]

m = 0
wee_list = []
for end in monthly_indeces:
    wee = np.corrcoef(obs[m * 30 : end], mod[m * 30 : end])[0, 1]
    m = m + 1
    wee_list.append(wee)

print(f"wee sum is {sum(wee_list)/len(wee_list)}")

f, (a0, a1) = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"width_ratios": [5, 1]})

a0.plot(obs, c="k", label="Observations")
a0.plot(mod, c="c", alpha=0.7, label="Model")
a0.legend(loc="upper right")


month_loc = [i * 360 for i in range(5)]
a0.set_xticks(month_loc, range(1, 6))
for i in month_loc:
    a0.axvline(x=i, color="k", alpha=0.2, zorder=-1)


a1.scatter(1, total, s=80, c="k", marker=(5, 1))
bp = a1.boxplot(wee_list, showmeans=True)

for mean in bp["means"]:
    mean.set_markerfacecolor("#000000")
    mean.set_markeredgecolor("#000000")
    mean.set_marker("D")

# Setting median line to black
for median in bp["medians"]:
    median.set_color("#FFFFFF")
a1.set_xticklabels([""])

f.tight_layout()
plt.savefig("/home/hma000/accomatic-web/plotting/out/pearson.png")
