import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path
import sys
import glob
from read_data import get_nc_point_data
from plot_precip import get_colour

print('hello')


dir = '/home/hma000/storage/yk_kdi_ldg/scaled/ykl/'
for pth in glob.glob(dir):
    data = os.path.basename(pth).split('_') # [1] + '_' + os.path.basename(pth).split('_')[2]
    print(data); sys.exit()
    df = get_nc_point_data(pth, site_dict)
    plt.plot(data[site], c=get_colour(pth))
    plt.xlabel('Time')
    plt.xticks(rotation=70)
plt.savefig(f'/home/hma000/accomatic-web/accomatic/merra2_precip/plots/13MAR/all_sites.png')
    