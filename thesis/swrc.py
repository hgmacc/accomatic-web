# These are the van Genuchten (1980) equations
# The input is matric potential, psi and the hydraulic parameters.
# psi must be sent in as a numpy array.
# The pars variable is like a MATLAB structure.
import math
import numpy as np


def thetaFun(psi, pars):
    Se = (1 + abs(psi * pars["AlphaVanGenuchten"]) ** pars["NVanGenuchten"]) ** (
        -pars["m"]
    )
    Se[psi >= 0] = 1.0
    return pars["ThetaRes"] + (pars["ThetaSat"] - pars["ThetaRes"]) * Se


def PlotProps(pars_l):
    import pylab as plt
    import swrc as vg

    fig, ax = plt.subplots(figsize=(9, 5))
    props = dict(boxstyle="round", facecolor="white")
    fc = 330

    plt.text(x=0, y=0.8, c="b", s="Wet\n0FC (0 hPa)", bbox=props)
    plt.axvline(x=0, color="b", linestyle="dashdot")

    plt.text(
        x=-math.log10(0.5 * fc), y=0.7, c="k", s="Mix\n0.5FC (165 hPa)\n", bbox=props
    )
    plt.axvline(x=-math.log10(0.5 * fc), c="k", linestyle="dashdot")

    plt.text(x=(-math.log10(2 * fc)), y=0.6, c="r", s="Dry\n2FC (660 hPa)", bbox=props)
    plt.axvline(x=-math.log10(2 * fc), color="r", linestyle="dashdot")

    psi = np.linspace(-10, 1, 100)
    for pars in pars_l:
        plt.plot(psi, vg.thetaFun(psi, pars), linewidth=2, label=pars["name"])
    plt.ylabel(r"$\theta(\psi)$")
    plt.xlabel(r"pF")
    plt.legend(
        loc="upper left",
        facecolor="white",
        frameAlphaVanGenuchten=1.0,
        edgecolor="white",
    )

    secax = ax.secondary_xaxis("top")
    secax.set_xlabel(r"$\psi [m]$ / hPa")
    locs, labels = plt.xticks()
    secax.set_xticks(locs, [f"10^{round(np.abs(i))}" for i in locs])
    plt.savefig("thesis/swrc.png")


def BeitNetofaClay():
    pars = {}
    pars["name"] = "BeitNetofaClay"
    pars["ThetaRes"] = 0.0
    pars["ThetaSat"] = 0.446
    pars["AlphaVanGenuchten"] = 0.152
    pars["NVanGenuchten"] = 1.17
    pars["m"] = 1 - 1 / pars["NVanGenuchten"]
    return pars


def peat():
    pars = {}
    pars["name"] = "peat"
    pars["ThetaRes"] = 0.2
    pars["ThetaSat"] = 0.85
    pars["AlphaVanGenuchten"] = 0.3  # 0.003
    pars["NVanGenuchten"] = 1.8
    pars["m"] = 1 - 1 / pars["NVanGenuchten"]
    pars["NormalHydrConductivity"] = 0.3
    pars["LateralHydrConductivity"] = 0.3
    pars["TfreezingSoil"] = -0.01
    pars["ThermalConductivitySoilSolids"] = 3.0
    return pars


def silt():
    pars = {}
    pars["name"] = "silt"
    pars["ThetaRes"] = 0.057
    pars["ThetaSat"] = 0.487
    pars["AlphaVanGenuchten"] = 0.1  # 0.001
    pars["NVanGenuchten"] = 1.6
    pars["m"] = 1 - 1 / pars["NVanGenuchten"]
    pars["NormalHydrConductivity"] = 0.0051
    pars["LateralHydrConductivity"] = 0.0051
    pars["TfreezingSoil"] = -0.1
    pars["ThermalConductivitySoilSolids"] = 2.5
    return pars


def rock():
    pars = {}
    pars["name"] = "rock"
    pars["ThetaRes"] = 0.002
    pars["ThetaSat"] = 0.05
    pars["AlphaVanGenuchten"] = 0.1  # 0.001
    pars["NVanGenuchten"] = 1.2
    pars["m"] = 1 - 1 / pars["NVanGenuchten"]
    pars["NormalHydrConductivity"] = 0.000001
    pars["LateralHydrConductivity"] = 0.000001
    pars["TfreezingSoil"] = 0
    pars["ThermalConductivitySoilSolids"] = 2.5
    return pars


def clay():
    pars = {}
    pars["name"] = "clay"
    pars["ThetaRes"] = 0.072
    pars["ThetaSat"] = 0.475
    pars["AlphaVanGenuchten"] = 0.1  # 0.001
    pars["NVanGenuchten"] = 1.4
    pars["m"] = 1 - 1 / pars["NVanGenuchten"]
    pars["NormalHydrConductivity"] = 0.0019
    pars["LateralHydrConductivity"] = 0.0019
    pars["TfreezingSoil"] = -0.05
    pars["ThermalConductivitySoilSolids"] = 2.5
    return pars


def sand():
    pars = {}
    pars["name"] = "sand"
    pars["ThetaRes"] = 0.055
    pars["ThetaSat"] = 0.374
    pars["AlphaVanGenuchten"] = 0.3  # 0.003
    pars["NVanGenuchten"] = 3.2
    pars["m"] = 1 - 1 / pars["NVanGenuchten"]
    pars["NormalHydrConductivity"] = 0.0825
    pars["LateralHydrConductivity"] = 0.0825
    pars["TfreezingSoil"] = -0.01
    pars["ThermalConductivitySoilSolids"] = 3.0
    return pars


# PlotProps(funcs)
funcs = [silt(), rock(), clay(), sand(), peat()]

slt = np.array([round((1.289**i)) for i in range(1, 35)])
depths = []
for i in range(len(slt)):
    depths.append(np.sum(slt[: i + 1]))

alt = [
    15,
    18,
    22,
    26,
    31,
    37,
    45,
    54,
    64,
    77,
    93,
    111,
    134,
    160,
    193,
    231,
    277,
    333,
    399,
    479,
    575,
    690,
    828,
    994,
    1192,
    1431,
    1717,
    2061,
    2700,
    3000,
    4000,
    5000,
]
print(alt)

import matplotlib.pyplot as plt

plt.scatter(range(len(alt)), [i / 1000 for i in alt])
plt.ylabel("Soil Layer Thickness")
plt.savefig("/home/hma000/accomatic-web/SoilLayerThicknesses.png")
l = [
    1,
    3,
    5,
    8,
    12,
    17,
    23,
    31,
    41,
    54,
    70,
    91,
    118,
    153,
    198,
    256,
    331,
    428,
    552,
    712,
    919,
    1185,
    1528,
    1971,
    2542,
    3277,
    4225,
    5447,
    7022,
]

print(len(l))
peat = peat()
silt = silt()
clay = clay()
rock = rock()
sand = sand()

var = "AlphaVanGenuchten"
# First M
# Every soil type
par_list = []
for i in alt[alt < 1000]:
    par_list.append(peat[var])
for i in alt[alt < 8000]:
    par_list.append(silt[var])
for i in alt[alt > 8000]:
    par_list.append(rock[var])
print(par_list)
print(len(par_list))
print(len(alt))
# M 2
# Peat turns into Silt

# M8
# Rock begins
