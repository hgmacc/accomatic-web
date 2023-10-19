import pandas as pd
import numpy as np


import matplotlib.pyplot as plt

slt = np.array([round((1.289**i)) for i in range(1, 35)])
print(np.sum(slt))


# plt.scatter(range(len(slt)), [i / 1000 for i in slt])
# plt.ylabel("Soil Layer Thickness")
# plt.savefig("/home/hma000/accomatic-web/SoilLayerThicknesses.png")

# First M
# Every soil type

# M 2
# Peat turns into Silt

# M8
# Rock begins
